# Installation

AutoTuneNet requires Python 3.10 or newer.

## From PyPI

```bash
pip install autotunenet                 # library + `autotunenet` command
pip install "autotunenet[bench]"        # + Ray Tune (PBT), Kaggle CLI, matplotlib (report figures)
```

| Install | Dependencies | Gives you |
|---|---|---|
| `autotunenet` | `optuna`, `torch`, `torchvision`, `pyyaml` | The library, and the benchmark harness for every method except PBT |
| `autotunenet[bench]` | adds `ray[tune]`, `kaggle`, `six`, `matplotlib` | PBT, `--runtime kaggle`, figures in `autotunenet report` |

PyTorch is installed from PyPI by default. For a specific CUDA build, install `torch` first by following [pytorch.org](https://pytorch.org/get-started/locally/), then install AutoTuneNet.

Check the installation:

```bash
python -c "import autotunenet; print('ok')"
autotunenet --help
```

## From source

For development, or to reproduce the published benchmarks:

```bash
git clone https://github.com/mitishraina/AutoTuneNet.git
cd AutoTuneNet
uv sync                 # .venv with the exact versions in uv.lock: package, [bench], dev tools
uv run pytest           # about 90 s on a laptop CPU
```

[uv](https://docs.astral.sh/uv/) reads `uv.lock`, so everyone gets identical versions. Inside the checkout, prefix commands with `uv run` (`uv run autotunenet list`). Without uv, `pip install -e ".[bench]"` also works but is not pinned.

| uv dependency group | Contents | Installed by |
|---|---|---|
| `dev` | `pytest`, `ruff`, `black`, and the `bench` extra | `uv sync` (default) |
| `docs` | `mkdocs-material`, `mkdocstrings` | `uv sync --group docs` |

## Devices

AutoTuneNet itself does no tensor work, so it runs wherever your model runs. The benchmark harness picks the device with `--device auto` (the default), in this order: CUDA, Apple MPS, CPU.
