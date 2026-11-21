import json

from autotunenet.bench.core.experiment import Experiment
from autotunenet.bench.runtimes import RUNTIMES, get_runtime
from autotunenet.bench.runtimes.kaggle import GPUS_PER_NOTEBOOK, Kaggle, render_notebook, slugify


def test_slugs_are_kaggle_safe():
    assert slugify("autotunenet Comparison 2 of 3") == "autotunenet-comparison-2-of-3"


def test_runtimes_are_registered():
    assert {"machine", "kaggle"} <= set(RUNTIMES)
    assert get_runtime("machine").name == "machine"


def test_notebook_has_readable_steps_and_valid_code():
    job = {"install": "autotunenet[bench]==1.0.2", "experiment": "pilot", "experiment_yaml": "name: pilot\n",
           "plugins": {}, "gpu_cells": [[{"method": "pbt", "task": "mnist", "scenario": "stationary", "seed": 0}], []]}
    nb = render_notebook(job, "autotunenet pilot 1 of 2")
    assert nb["nbformat"] == 4
    kinds = [c["cell_type"] for c in nb["cells"]]
    assert kinds[0] == "markdown" and kinds.count("code") >= 4
    title = "".join(nb["cells"][0]["source"])
    assert "autotunenet pilot 1 of 2" in title and "autotunenet[bench]==1.0.2" in title
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    compile(code, "run.ipynb", "exec")
    namespace = {}
    exec(code.split("WORK =")[0], namespace)
    assert namespace["JOB"] == job


def test_prepare_spreads_every_pending_cell_over_the_gpus_once(tmp_path, monkeypatch):
    monkeypatch.setattr(Kaggle, "user", property(lambda self: "someone"))
    monkeypatch.setattr(Kaggle, "_dataset_exists", lambda self, task: True)
    exp = Experiment.load("smoke")
    from autotunenet.bench.core.experiment import resolve_experiment
    Kaggle().prepare(exp, resolve_experiment("smoke"), exp.cells(), tmp_path, [], parallel=2)

    state = json.loads((tmp_path / "smoke" / ".runtime" / "kaggle.json").read_text())
    assert state["round"] == 1 and len(state["notebooks"]) <= 2
    assigned = []
    for nb in state["notebooks"]:
        notebook = json.loads((tmp_path / "smoke" / "notebooks" / f"{nb['slug']}.ipynb").read_text())
        code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
        ns = {}
        exec(code.split("WORK =")[0], ns)
        assert len(ns["JOB"]["gpu_cells"]) == GPUS_PER_NOTEBOOK
        assigned += [tuple(c.values()) for g in ns["JOB"]["gpu_cells"] for c in g]
    assert sorted(assigned) == sorted(tuple(c.to_dict().values()) for c in exp.cells())
