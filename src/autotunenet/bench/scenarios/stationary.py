from autotunenet.bench.core.registry import SCENARIOS
from autotunenet.bench.scenarios.base import Scenario


@SCENARIOS.register("stationary")
class Stationary(Scenario):
    """Plain training: nothing changes during the run."""
