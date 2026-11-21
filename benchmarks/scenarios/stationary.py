from benchmarks.core.registry import SCENARIOS
from benchmarks.scenarios.base import Scenario


@SCENARIOS.register("stationary")
class Stationary(Scenario):
    """Plain training: nothing changes during the run."""
