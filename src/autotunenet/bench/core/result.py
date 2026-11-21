"""The single result schema every method produces, and where it is stored."""
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1


@dataclass
class RunOutcome:
    """What a method returns: test metrics, per-epoch history, method extras."""
    test_loss: float
    test_accuracy: float
    history: List[Dict[str, Any]]        # one dict per epoch of the reported model
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    experiment: str
    method: str
    task: str
    scenario: str
    seed: int
    test_loss: float
    test_accuracy: float
    history: List[Dict[str, Any]]
    shift_epoch: Optional[int]
    compute: Dict[str, Any]
    environment: Dict[str, Any]
    config: Dict[str, Any]
    extra: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def save(self, path: Path) -> None:
        """Write atomically, so an interrupted run never leaves a half-written file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(asdict(self), f, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "RunResult":
        with open(path) as f:
            return cls(**json.load(f))


def result_path(root: Path, experiment: str, method: str, task: str, scenario: str, seed: int) -> Path:
    return Path(root) / experiment / method / task / scenario / f"seed_{seed}.json"
