"""AutoTuneNet on stationary training. Usage: python -m benchmarks.autotunenet.run --task mnist|cifar10 --seed 42"""
import torch

from benchmarks.common.legacy import LegacyRun, autotunenet_config
from benchmarks.common.train_utils import train_model


def main():
    run = LegacyRun(__file__, "AutoTuneNet on stationary training")
    lr = run.defaults["lr"]

    model = run.new_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    adapter = run.autotunenet_adapter(optimizer)
    lr_history, rollbacks = [], 0

    def on_epoch_end(epoch, val_loss, val_accuracy):
        nonlocal rollbacks
        adapter.on_epoch_end(metric=-val_loss)
        rollbacks += int(adapter.last_rollback)
        lr_history.append(optimizer.param_groups[0]["lr"])

    history = train_model(
        model, run.train_loader, run.val_loader, optimizer, run.device, run.epochs,
        on_epoch_end=on_epoch_end,
    )

    test = run.test_metrics(model)
    run.save("metrics.json", {**history, **test, "epochs_trained": run.epochs, "rollbacks": rollbacks})
    run.save("lr_history.json", lr_history)
    run.save("run_info.json", run.run_info(
        method="autotunenet", initial_lr=lr, autotunenet=autotunenet_config(run.task)
    ))
    print(f"AutoTuneNet: test_accuracy={test['test_accuracy']:.4f} test_loss={test['test_loss']:.4f} rollbacks={rollbacks}")


if __name__ == "__main__":
    main()
