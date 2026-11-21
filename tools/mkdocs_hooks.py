"""MkDocs hook: publish the benchmark figures next to the pages that show them.

docs/benchmarks/results.md links images as ../../benchmarks/results/..., which
resolves inside the repository on GitHub. Copying those files to the same path
in the built site makes the same links work on the documentation site.
"""
import shutil
from pathlib import Path

SOURCE = Path("benchmarks/results")
SUFFIXES = {".png", ".svg"}


def on_post_build(config, **kwargs):
    site = Path(config["site_dir"])
    for path in SOURCE.rglob("*"):
        if path.suffix in SUFFIXES and path.is_file():
            target = site / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
