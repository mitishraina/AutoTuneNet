# Contributing to AutoTuneNet

<!-- --8<-- [start:body] -->
Bug reports, benchmark tasks and scenarios, comparison methods and framework adapters are all welcome. For a larger change, open an issue first so the design can be agreed before you write code.

## Development setup

```bash
git clone https://github.com/mitishraina/AutoTuneNet.git
cd AutoTuneNet
uv sync                     # package, [bench] extra and dev tools, pinned by uv.lock
uv run pytest               # about 90 s on a laptop CPU
```

| Task | Command |
|---|---|
| Run the tests | `uv run pytest` |
| Run one test file | `uv run pytest tests/test_stability.py -q` |
| Check every benchmark method and scenario end to end | `uv run autotunenet run --experiment smoke --device cpu` |
| Preview the documentation site | `uv run --group docs mkdocs serve` |
| Build it as CI does | `uv run --group docs mkdocs build --strict` |
| Add a dependency | `uv add <pkg>` (library), `uv add --optional bench <pkg>`, `uv add --group dev <pkg>` |

Commit `uv.lock` whenever `pyproject.toml` changes; CI installs with `uv sync --locked` and fails if the two disagree.

## Where things go

| Change | Location | Also update |
|---|---|---|
| Controller behaviour | `src/autotunenet/` | Tests in `tests/`, [How it works](https://mitishraina.github.io/AutoTuneNet/concepts/) if behaviour changes |
| A benchmark method | `src/autotunenet/bench/methods/`, imported in its `__init__.py` | `experiments/smoke.yaml` (the end-to-end test fails if a method is missing) |
| A benchmark task or scenario | `src/autotunenet/bench/tasks/` or `scenarios/` | `smoke.yaml` for scenarios |
| A runtime | `src/autotunenet/bench/runtimes/`, registered in `RUNTIMES` | `docs/benchmarks/runtimes.md` |
| A command or option | `src/autotunenet/cli.py` (`COMMANDS` holds the help text) | `docs/benchmarks/cli.md` |
| Public API | docstrings (Google style) | Rendered automatically in the API reference |

## Pull requests

- Branch from `main`, named by kind: `feat/…`, `fix/…`, `docs/…`, `chore/…`, `refactor/…`.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/): `fix(adapter): restore the rollback target exactly`.
- New behaviour comes with tests, and a fixed bug comes with a test that failed before the fix.
- Add a line under `Unreleased` in `CHANGELOG.md` for anything a user would notice.
- CI runs the tests on Python 3.10 and 3.11 and builds the documentation; both must pass.

## Benchmark results

Results under `benchmarks/results/` are evidence, so treat them as such:

- Do not edit result files by hand. Regenerate them with `autotunenet run`.
- Published runs keep the notebooks or environment that produced them. Each result records its hardware, library versions and git commit.
- Compare methods only on the same hardware type; `autotunenet report` warns when a comparison mixes them.
- Report mean ± std over seeds, the compute each method used, and results where AutoTuneNet loses.
<!-- --8<-- [end:body] -->
