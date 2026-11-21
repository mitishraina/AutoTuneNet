# Examples

Each script runs on a CPU in seconds (except `bench_plugin.py`, which downloads Fashion-MNIST).

| File | Shows | Run |
|---|---|---|
| [`quickstart.py`](quickstart.py) | Tuning `lr` once per epoch in a plain training loop | `python examples/quickstart.py` |
| [`multi_param.py`](multi_param.py) | Tuning `lr`, `momentum` and `weight_decay` together; two param groups keep their 10× ratio; custom stability settings | `python examples/multi_param.py` |
| [`from_config.py`](from_config.py) + [`autotune.yaml`](autotune.yaml) | Building the adapter from a YAML config | `python examples/from_config.py` |
| [`bench_plugin.py`](bench_plugin.py) + [`bench_experiment.yaml`](bench_experiment.yaml) | A benchmark plugin: your own task (dataset + model) and scenario, compared across methods | `autotunenet run --experiment examples/bench_experiment.yaml --plugins examples/bench_plugin.py` |

Explanations: [Quick start](https://mitishraina.github.io/AutoTuneNet/quickstart/), [Configuration](https://mitishraina.github.io/AutoTuneNet/configuration/), [Extending the benchmark](https://mitishraina.github.io/AutoTuneNet/benchmarks/extending/).
