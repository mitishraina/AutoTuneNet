from autotunenet_bench.controllers.autotunenet import AutoTuneNetController
from autotunenet_bench.controllers.fixed_lr import FixedLRController
from autotunenet_bench.controllers.scheduler import SchedulerController


def build_controller(
    cfg: dict,
    optimizer,
    adapter=None,
    scheduler=None,
):
    kind = cfg["type"]

    if kind == "autotunenet":
        if adapter is None:
            raise ValueError("AutoTuneNetController requires adapter")
        return AutoTuneNetController(adapter)

    if kind == "fixed_lr":
        if "lr" in cfg:
            lr = cfg["lr"]
        elif "parameter_space" in cfg and "lr" in cfg["parameter_space"]:
            lr = cfg["parameter_space"]["lr"][0]
        else:
            raise ValueError("fixed_lr requires lr or parameter_space.lr")

        return FixedLRController(lr=lr)

    if kind == "scheduler":
        if scheduler is None:
            raise ValueError("SchedulerController requires scheduler")
        return SchedulerController(scheduler)

    raise ValueError(f"Unknown controller type: {kind}")
