import argparse
import torch
from pathlib import Path

from autotunenet_bench.regimes.factory import build_regime
from autotunenet_bench.configs.loader import load_benchmark_config
from autotunenet_bench.runners.train_runner import train

from autotunenet.parameters import ParameterSpace
from autotunenet.bayesian_optimizer import BayesianOptimizer
from autotunenet.adapters.pytorch.adapter import PyTorchHyperParameterAdapter

from benchmarks.common.dataset import get_mnist_loaders
from benchmarks.common.model import CNN

from autotunenet_bench.controllers.factory import build_controller

def parse_args():
    parser = argparse.ArgumentParser("AutoTuneNet Benchmark Runner")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to benchmark config YAML"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_benchmark_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    run_name = Path(args.config).stem
    output_dir = Path("autotunenet_bench/results") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)


    train_loader, val_loader = get_mnist_loaders(
        batch_size=config["training"]["batch_size"],
        seed=seed
    )

    model = CNN()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["initial_lr"]
    )

    controller_cfg = config["controller"]

    param_space = ParameterSpace(controller_cfg["parameter_space"])

    autotune = BayesianOptimizer(
        param_space=param_space,
        seed=seed,
        smoothing_window=controller_cfg.get("metrics", {}).get(
            "smoothing_window", 5
        )
    )

    adapter = PyTorchHyperParameterAdapter(
        torch_optimizer=optimizer,
        autotune_optimizer=autotune,
        tune_n_steps=controller_cfg.get("tuning", {}).get("tune_n_steps", 1),
        warmup_epochs=controller_cfg.get("tuning", {}).get("warmup_epochs", 0),
        warmup_metric_threshold=controller_cfg.get("tuning", {}).get(
            "warmup_metric_threshold"
        ),
        max_delta=controller_cfg.get("tuning", {}).get("max_delta"),
    )
    scheduler=None
    if controller_cfg["type"] == "scheduler":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["epochs"],
        )
    controller = build_controller(
        cfg=controller_cfg,
        optimizer=optimizer,
        adapter=adapter,
        scheduler=scheduler
    )


    regime = build_regime(config.get("non_stationarity"))

    train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        controller=controller,
        regime=regime,
        epochs=config["training"]["epochs"],
        device=device,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
