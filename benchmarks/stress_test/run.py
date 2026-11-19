"""
Stress test: every method starts from a deliberately bad learning rate (0.1).
Usage: python -m benchmarks.stress_test.run --task mnist|cifar10 --seed 42
"""
import torch

from benchmarks.common.legacy import LegacyRun, autotunenet_config
from benchmarks.common.train_utils import train_model

BAD_LR = 0.1


def run_fixed(run):
    model = run.new_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=BAD_LR)
    history = train_model(model, run.train_loader, run.val_loader, optimizer, run.device, run.epochs)
    return model, history, [BAD_LR] * run.epochs, 0


def run_scheduler(run):
    model = run.new_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=BAD_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=run.epochs)
    lrs = []

    def on_epoch_end(epoch, val_loss, val_accuracy):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    history = train_model(model, run.train_loader, run.val_loader, optimizer, run.device, run.epochs,
                          on_epoch_end=on_epoch_end)
    return model, history, lrs, 0


def run_autotunenet(run):
    model = run.new_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=BAD_LR)
    adapter = run.autotunenet_adapter(optimizer)
    lrs, rollbacks = [], 0

    def on_epoch_end(epoch, val_loss, val_accuracy):
        nonlocal rollbacks
        adapter.on_epoch_end(metric=-val_loss)
        rollbacks += int(adapter.last_rollback)
        lrs.append(optimizer.param_groups[0]["lr"])

    history = train_model(model, run.train_loader, run.val_loader, optimizer, run.device, run.epochs,
                          on_epoch_end=on_epoch_end)
    return model, history, lrs, rollbacks


def main():
    run = LegacyRun(__file__, "Stress test with a bad initial learning rate")
    results = {}
    for name, fn in [("fixed_lr", run_fixed), ("scheduler", run_scheduler), ("autotunenet", run_autotunenet)]:
        model, history, lrs, rollbacks = fn(run)
        results[name] = {**history, **run.test_metrics(model), "lr_history": lrs,
                         "rollbacks": rollbacks, "epochs_trained": run.epochs}
        print(f"{name}: test_accuracy={results[name]['test_accuracy']:.4f}")

    run.save("results.json", results)
    run.save("run_info.json", run.run_info(
        method="stress_test", initial_lr=BAD_LR, autotunenet=autotunenet_config(run.task)
    ))


if __name__ == "__main__":
    main()
