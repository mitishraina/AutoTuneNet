"""Where results and datasets live. Both can be overridden by environment variables."""
import os
from pathlib import Path


def results_dir() -> Path:
    """
    $AUTOTUNENET_RESULTS if set; inside the AutoTuneNet repository the tracked
    benchmarks/results/; otherwise ./autotunenet_results in the current folder.
    """
    if os.environ.get("AUTOTUNENET_RESULTS"):
        return Path(os.environ["AUTOTUNENET_RESULTS"])
    if Path("benchmarks/results").is_dir():
        return Path("benchmarks/results")
    return Path("autotunenet_results")


def data_dir() -> Path:
    """$AUTOTUNENET_DATA_DIR if set, otherwise ./data in the current folder."""
    return Path(os.environ.get("AUTOTUNENET_DATA_DIR") or Path.cwd() / "data")
