"""Load user code (tasks, models, scenarios, methods) that registers itself."""
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable

ENV = "AUTOTUNENET_PLUGINS"    # loaded plugins, for worker processes (e.g. Ray) to load again


def load_user_plugins(specs: Iterable[str]) -> None:
    """
    Each spec is a Python file (my_tasks.py) or an importable module
    (mypackage.tasks). Importing it runs its @TASKS / @SCENARIOS / @METHODS
    registrations.
    """
    loaded = [s for s in os.environ.get(ENV, "").split(os.pathsep) if s]
    for spec in [s for s in (specs or []) if s]:
        path = Path(spec)
        if path.suffix == ".py":
            if not path.exists():
                raise SystemExit(f"Plugin file not found: {spec}")
            spec = str(path.resolve())
            name = f"autotunenet_plugin_{path.stem}"
            if name in sys.modules:
                continue
            module_spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[name] = module
            module_spec.loader.exec_module(module)
        else:
            importlib.import_module(spec)
        if spec not in loaded:
            loaded.append(spec)
    os.environ[ENV] = os.pathsep.join(loaded)


def reload_in_worker() -> None:
    """Register the built-ins and the parent process's plugins in a fresh worker process."""
    from autotunenet.bench.core.registry import load_plugins
    load_plugins()
    load_user_plugins(os.environ.get(ENV, "").split(os.pathsep))
