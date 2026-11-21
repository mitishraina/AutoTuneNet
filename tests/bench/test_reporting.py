import xml.dom.minidom

from benchmarks import report, run
from benchmarks.reporting.matrix import render_markdown, render_svg
from benchmarks.reporting.table import build_matrix


def _group(method, task, scenario, acc, epochs=10, parallel=1, restarts=0):
    stat = lambda v: {"mean": v, "std": 0.0, "n": 3, "values": [v] * 3}  # noqa: E731
    return {"method": method, "task": task, "scenario": scenario, "metrics": {
        "test_accuracy": stat(acc), "epochs_trained": stat(epochs),
        "max_parallel_models": stat(parallel), "trials": stat(1 + restarts), "restarts": stat(restarts)}}


def _summary(*groups):
    return {"groups": {f"{g['method']}/{g['task']}/{g['scenario']}": g for g in groups}}


def test_ties_share_a_rank_and_worse_than_baseline_is_flagged():
    m = build_matrix(_summary(
        _group("autotunenet", "mnist", "stationary", 0.95),
        _group("cosine", "mnist", "stationary", 0.97),
        _group("plateau", "mnist", "stationary", 0.97),
        _group("fixed_lr", "mnist", "stationary", 0.96),
    ))
    assert m.cell("cosine", "mnist", "stationary").rank == 1
    assert m.cell("plateau", "mnist", "stationary").rank == 1
    assert m.cell("fixed_lr", "mnist", "stationary").rank == 3
    assert m.cell("autotunenet", "mnist", "stationary").rank == 4
    assert m.cell("autotunenet", "mnist", "stationary").worse_than_baseline
    assert not m.cell("cosine", "mnist", "stationary").worse_than_baseline
    assert m.rows[-1] == "fixed_lr"                       # baseline row last


def test_missing_cells_are_not_applicable():
    m = build_matrix(_summary(
        _group("fixed_lr", "mnist", "bad_initial_lr", 0.10),
        _group("asha", "mnist", "stationary", 0.99, epochs=48, restarts=15),
        _group("fixed_lr", "mnist", "stationary", 0.96),
    ))
    assert m.cell("asha", "mnist", "bad_initial_lr").mean is None
    assert "n/a" in render_markdown(m)
    assert "n/a" in render_svg(m, "t")
    assert m.cost["asha"] == {"epochs": 48, "parallel": 1, "restarts": 15}


def test_report_end_to_end_on_the_smoke_run(tmp_path):
    assert run.main(["--experiment", "benchmarks/experiments/smoke.yaml", "--results-dir", str(tmp_path),
                     "--device", "cpu", "--method", "fixed_lr", "--method", "cosine",
                     "--method", "autotunenet"]) == 0
    out = report.build_report(tmp_path, "smoke", "Support matrix")
    svg = (out / "support_matrix.svg").read_text()
    xml.dom.minidom.parseString(svg)                      # well-formed SVG
    assert "AutoTuneNet" in svg and "Baseline" in svg
    markdown = (out / "support_matrix.md").read_text()
    assert markdown.startswith("## Support matrix")
    assert "| Method |" in markdown
    assert "<svg" in (out / "support_matrix.html").read_text()
    for name in ("fig1_val_loss", "fig2_val_accuracy", "fig3_learning_rate", "fig4_stress_test"):
        assert (out / "paper_figures" / f"toy_{name}.png").exists()
    assert (out / "accuracy_vs_compute" / "toy_label_noise.png").exists()
    assert (out / "curves" / "toy_label_noise_val_loss.png").exists()
