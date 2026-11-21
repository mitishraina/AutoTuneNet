"""Cosine-annealing scheduler baseline. Usage: python -m benchmarks.scheduler.run --task mnist|cifar10 --seed 42"""
import torch

from benchmarks.common.legacy import LegacyRun
from benchmarks.common.train_utils import train_model


def main():
    run = LegacyRun(__file__, "Cosine-annealing scheduler baseline")
    lr = run.defaults["lr"]

    model = run.new_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=run.epochs)
    lr_history = []

    def on_epoch_end(epoch, val_loss, val_accuracy):
        scheduler.step()
        lr_history.append(optimizer.param_groups[0]["lr"])

    history = train_model(
        model, run.train_loader, run.val_loader, optimizer, run.device, run.epochs,
        on_epoch_end=on_epoch_end,
    )

    test = run.test_metrics(model)
    run.save("metrics.json", {**history, **test, "epochs_trained": run.epochs})
    run.save("lr_history.json", lr_history)
    run.save("run_info.json", run.run_info(method="scheduler", initial_lr=lr, scheduler="CosineAnnealingLR"))
    print(f"Scheduler: test_accuracy={test['test_accuracy']:.4f} test_loss={test['test_loss']:.4f}")


if __name__ == "__main__":
    main()
