import pytest

from autotunenet.bench.core.registry import load_plugins


@pytest.fixture(autouse=True, scope="session")
def _plugins():
    load_plugins()
