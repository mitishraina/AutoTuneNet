import json

from benchmarks.cloud.kaggle import GPUS_PER_NOTEBOOK, JOB_TEMPLATE, slugify


def test_slugs_are_kaggle_safe():
    assert slugify("autotunenet Comparison 2 of 3") == "autotunenet-comparison-2-of-3"


def test_job_template_renders_to_valid_python_with_the_job():
    job = {"repo": "https://github.com/x/y.git", "ref": "abc123", "experiment": "benchmarks/experiments/pilot.yaml",
           "shards": ["1/4", "2/4"], "filters": ["--method", "pbt"]}
    source = JOB_TEMPLATE.read_text().replace("__JOB__", json.dumps(job))
    compile(source, "run.py", "exec")
    namespace = {}
    exec(source.split("WORK =")[0], namespace)          # module header up to the JOB constant
    assert namespace["JOB"] == job


def test_every_shard_is_assigned_exactly_once():
    notebooks = 3
    shards = notebooks * GPUS_PER_NOTEBOOK
    assigned = [f"{k * GPUS_PER_NOTEBOOK + g + 1}/{shards}" for k in range(notebooks) for g in range(GPUS_PER_NOTEBOOK)]
    assert assigned == [f"{i}/{shards}" for i in range(1, shards + 1)]
