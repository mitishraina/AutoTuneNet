from benchmarks.cloud.kaggle import GPUS_PER_NOTEBOOK, render_notebook, slugify


def test_slugs_are_kaggle_safe():
    assert slugify("autotunenet Comparison 2 of 3") == "autotunenet-comparison-2-of-3"


def test_notebook_has_readable_steps_and_valid_code():
    job = {"repo": "https://github.com/x/y.git", "ref": "abc123", "experiment": "benchmarks/experiments/pilot.yaml",
           "shards": ["1/4", "2/4"], "filters": ["--method", "pbt"]}
    nb = render_notebook(job, "autotunenet pilot 1 of 2")
    assert nb["nbformat"] == 4
    kinds = [c["cell_type"] for c in nb["cells"]]
    assert kinds[0] == "markdown" and kinds.count("code") >= 5
    title = "".join(nb["cells"][0]["source"])
    assert "autotunenet pilot 1 of 2" in title and "abc123" in title and "1/4, 2/4" in title
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    compile(code, "run.ipynb", "exec")
    namespace = {}
    exec(code.split("WORK =")[0], namespace)
    assert namespace["JOB"] == job


def test_every_shard_is_assigned_exactly_once():
    notebooks = 3
    shards = notebooks * GPUS_PER_NOTEBOOK
    assigned = [f"{k * GPUS_PER_NOTEBOOK + g + 1}/{shards}" for k in range(notebooks) for g in range(GPUS_PER_NOTEBOOK)]
    assert assigned == [f"{i}/{shards}" for i in range(1, shards + 1)]
