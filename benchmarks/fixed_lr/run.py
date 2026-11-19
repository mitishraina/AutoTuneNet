"""Fixed learning-rate baseline. Usage: python -m benchmarks.fixed_lr.run --task mnist|cifar10 --seed 42"""
import torch

from benchmarks.common.legacy import LegacyRun
from benchmarks.common.train_utils import train_model


def main():
    run = LegacyRun(__file__, "Fixed learning-rate baseline")
    lr = run.defaults["lr"]

    model = run.new_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = train_model(model, run.train_loader, run.val_loader, optimizer, run.device, run.epochs)

    test = run.test_metrics(model)
    run.save("metrics.json", {**history, **test, "epochs_trained": run.epochs})
    run.save("run_info.json", run.run_info(method="fixed_lr", lr=lr, optimizer="Adam"))
    print(f"Fixed LR ({lr}): test_accuracy={test['test_accuracy']:.4f} test_loss={test['test_loss']:.4f}")


if __name__ == "__main__":
    main()
