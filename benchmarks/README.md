# benchmarks/

Published benchmark results. The harness itself is part of the package (`autotunenet.bench`) and runs with the `autotunenet` command; see the [benchmark documentation](https://mitishraina.github.io/AutoTuneNet/benchmarks/).

| Path | Contents |
|---|---|
| [`results/legacy/`](results/legacy/) | Raw output of the earlier benchmark scripts (Apple MPS, 3 seeds), unchanged |
| [`results/previous/`](results/previous/) | `legacy/` converted to the current `RunResult` format, with its report; shown in the main README |
| [`results/pilot/`](results/pilot/) | Small Kaggle timing run, with the notebook that ran it |
| `results/comparison/` | The full comparison, with its Kaggle notebooks (once run) |
| [`legacy_import.py`](legacy_import.py) | Regenerates `previous/` from `legacy/`: `uv run python benchmarks/legacy_import.py` |

Each experiment folder has one JSON per run at `<method>/<task>/<scenario>/seed_<s>.json`, a `report/` with the support matrix and figures, and for Kaggle runs a `notebooks/` folder. How to verify a result: [Results](https://mitishraina.github.io/AutoTuneNet/benchmarks/results/).

Other experiments you run in this repository are git-ignored, so only published runs are tracked.
