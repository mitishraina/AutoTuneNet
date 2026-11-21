"""Display names and orderings shared by every report artefact."""

METHOD_LABELS = {
    "autotunenet": "AutoTuneNet",
    "cosine": "Cosine schedule",
    "plateau": "ReduceLROnPlateau",
    "optuna_tpe": "Optuna TPE",
    "asha": "ASHA",
    "hyperband": "Hyperband",
    "pbt": "PBT",
    "fixed_lr": "Fixed LR",
}
TASK_LABELS = {"mnist": "MNIST", "cifar10": "CIFAR-10", "toy": "Toy"}
SCENARIO_LABELS = {
    "stationary": "stationary",
    "label_noise": "label noise",
    "gradient_noise": "gradient noise",
    "bad_initial_lr": "bad initial LR",
}
METHOD_ORDER = list(METHOD_LABELS)          # fixed_lr last: it is the baseline row
SCENARIO_ORDER = list(SCENARIO_LABELS)
BASELINE = "fixed_lr"


def label(mapping, key):
    return mapping.get(key, key)


def ordered(keys, order):
    """Known keys in the preferred order, unknown ones after them alphabetically."""
    known = [k for k in order if k in keys]
    return known + sorted(k for k in keys if k not in order)
