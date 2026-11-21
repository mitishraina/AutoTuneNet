# Runtimes

A runtime is where the cells of an experiment train. Choose one with `--runtime`; without it, cells train on this machine.

| Runtime | `--runtime` | Needs | Parallelism |
|---|---|---|---|
| This machine | *(omit)* | nothing | one cell at a time, on the best device (CUDA → MPS → CPU) |
| Kaggle | `kaggle` | a Kaggle account with a verified phone and an API token | `--parallel` notebooks × 2 T4 GPUs |

## How resuming works

Every cell writes its result to `<results>/<experiment>/<method>/<task>/<scenario>/seed_<s>.json` as soon as it finishes. Every command first lists the cells that have no result yet and works only on those. So you can stop any command at any point (Ctrl-C, a closed laptop, a timed-out notebook) and rerun it to continue. Finished cells are never run twice. Results from several machines merge by copying the folders together.

| Location | Default |
|---|---|
| Results | `benchmarks/results` inside this repository, `./autotunenet_results` anywhere else, or `$AUTOTUNENET_RESULTS` |
| Datasets | `./data`, or `$AUTOTUNENET_DATA_DIR` |

## This machine

```bash
autotunenet run --experiment comparison
autotunenet run --experiment comparison --device cpu --method autotunenet --task mnist
```

Datasets download on first use, with a progress bar. To fetch them up front: `autotunenet data --experiment comparison`.

## Kaggle

Kaggle gives free GPU time ("GPU T4 x2", about 30 hours a week). Everything runs under **your** account; no credentials are stored in the repository or in the notebooks.

1. Create a token: kaggle.com → your profile picture → **Settings** → **API** → **Create New Token**, and save it as `~/.kaggle/access_token` (or a downloaded `kaggle.json` in `~/.kaggle/`), then `chmod 600` it. Never commit it or paste it anywhere.
2. Run one command:

```bash
autotunenet run --experiment comparison --runtime kaggle --parallel 3
```

It checks the Kaggle CLI and token, uploads each task's data once as a private dataset `autotunenet-<task>`, writes the notebooks for the cells without results (two GPUs each, `--parallel` notebooks at once), starts them, prints progress every minute, downloads each notebook's results as it finishes and prints the support matrix. Stop it whenever you like: running it again follows the notebooks still running, and once they are done plans a new round with whatever is still missing (failed or timed-out cells).

The same stages one at a time:

```bash
autotunenet setup   --runtime kaggle
autotunenet data    --runtime kaggle --experiment comparison   # --force to upload again
autotunenet prepare --runtime kaggle --experiment comparison --parallel 3
autotunenet start   --runtime kaggle --experiment comparison
autotunenet status  --runtime kaggle --experiment comparison
autotunenet pull    --runtime kaggle --experiment comparison
autotunenet report  --experiment comparison
```

The notebooks install AutoTuneNet from the exact commit you are on when it has been pushed (otherwise the released version), so push first if you changed the harness. They are saved as readable `.ipynb` files, one cell per step, under `<results>/<experiment>/notebooks/`. For the published experiments in this repository they are committed next to the results, so anyone can check what ran; your own experiments, their notebooks and the runtime state (`.runtime/`, which contains your username) are git-ignored or live outside the repository.
### What a Kaggle round looks like

1. **`setup`** checks the `kaggle` CLI and that a token exists in `~/.kaggle/`. The token is never read or printed.
2. **`data`** fetches each task's files locally (the `data_paths` of its `TaskSpec`). It packs them into `data.tar` with a marker file and creates a private dataset `<you>/autotunenet-<task>`. Datasets that already exist are skipped unless you pass `--force`.
3. **`prepare`** lists the pending cells and deals them round-robin across `--parallel × 2` GPU slots. It writes one notebook per pair of slots and records the round in `<results>/<experiment>/.runtime/kaggle.json`.
4. **`start`** pushes each notebook as a private kernel with GPU and internet enabled and the task datasets attached.
5. Inside each notebook, `pip install` installs the same AutoTuneNet commit, the data is linked into place, and `autotunenet run --cells ...` runs once per GPU with `CUDA_VISIBLE_DEVICES` set. The results and logs are packed into `autotunenet_results.tar.gz`.
6. **`pull`** downloads each finished notebook's output and adds the result files it does not have yet. Logs go to `.runtime/logs/`.

`run` does all of this and polls once a minute. A round ends when every notebook has been pulled; the next `run` plans a new round from whatever is still missing.

## Adding a runtime

A runtime subclasses `autotunenet.bench.runtimes.base.Runtime` and is registered in `autotunenet.bench.runtimes.RUNTIMES`. It implements:

| Method | Responsibility |
|---|---|
| `setup()` | Return a list of problems (empty = ready) |
| `data(tasks, force)` | Make each task's data available to the runtime |
| `prepare(exp, exp_path, cells, results_dir, plugins, parallel)` | Plan the pending cells |
| `start(exp, results_dir)` | Launch what was planned |
| `pull(exp, results_dir)` | Fetch finished results; return how many are new |
| `in_progress(exp, results_dir)`, `describe(exp, results_dir)` | Status for `autotunenet status` |
| `run(...)` | All stages; return the cells still unfinished |

Keep any state under `<results>/<experiment>/.runtime/<name>.json`, which is git-ignored. See [`runtimes/kaggle.py`](https://github.com/mitishraina/AutoTuneNet/blob/main/src/autotunenet/bench/runtimes/kaggle.py) for a complete example.
