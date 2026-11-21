"""
Offline HPO baseline: Optuna TPE over full (short) training runs, then a final
run at the best learning rate. Trials are selected on validation loss; the
final result is reported on the held-out test split.

Usage: python -m benchmarks.offline_hpo.run --task mnist|cifar10 --seed 42
"""
import optuna
import torch

from benchmarks.common.legacy import LegacyRun
from benchmarks.common.train_utils import train_model

N_TRIALS = 8
TRIAL_EPOCHS = 3


def main():
    run = LegacyRun(__file__, "Offline HPO baseline")
    low, high = run.defaults["lr_range"]

    def objective(trial):
        lr = trial.suggest_float("lr", low, high, log=True)
        model = run.new_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        history = train_model(model, run.train_loader, run.val_loader, optimizer, run.device, TRIAL_EPOCHS)
        return history["val_loss"][-1]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=run.seed))
    study.optimize(objective, n_trials=N_TRIALS)
    best_lr = study.best_params["lr"]

    model = run.new_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    history = train_model(model, run.train_loader, run.val_loader, optimizer, run.device, run.epochs)

    epochs_trained = N_TRIALS * TRIAL_EPOCHS + run.epochs
    test = run.test_metrics(model)
    run.save("metrics.json", {**history, **test, "epochs_trained": epochs_trained, "restarts": N_TRIALS})
    run.save("best_params.json", study.best_params)
    run.save("run_info.json", run.run_info(
        method="offline_hpo", n_trials=N_TRIALS, trial_epochs=TRIAL_EPOCHS,
        search_space={"lr": [low, high], "log": True},
    ))
    print(f"Offline HPO: best lr={best_lr:.6f} test_accuracy={test['test_accuracy']:.4f} epochs_trained={epochs_trained}")


if __name__ == "__main__":
    main()
