import argparse
import json
import torch
from pathlib import Path

from autotunenet_bench.regimes.factory import build_regime
from autotunenet_bench.configs.loader import load_benchmark_config
from autotunenet_bench.runners.train_runner import train

from autotunenet.factory import build_pytorch_autotunenet

from benchmarks.common.tasks import build_task, resolve_task

from autotunenet_bench.controllers.factory import build_controller

def parse_args():
    parser = argparse.ArgumentParser("AutoTuneNet Benchmark Runner")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to benchmark config YAML"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to train on. 'auto' picks CUDA, then Apple MPS, then CPU. "
             "Compare runs only when they used the same device.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but Apple MPS is not available")
    return torch.device(name)


def main():
    args = parse_args()
    config = load_benchmark_config(args.config)

    device = resolve_device(args.device)
    seed = config.get("seed", config["training"].get("seed", 42))
    torch.manual_seed(seed)

    run_name = Path(args.config).stem
    output_dir = Path("autotunenet_bench/results") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Task is read from the config; older configs without `task` are inferred from the file name.
    task = resolve_task(
        config.get("task") or ("cifar10" if "cifar" in run_name.lower() else "mnist")
    )
    print(f"[bench] run={run_name} task={task} seed={seed} device={device}")

    train_loader, val_loader, model = build_task(
        task,
        batch_size=config["training"]["batch_size"],
        seed=seed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["initial_lr"]
    )

    controller_cfg = config["controller"]

    # Only the AutoTuneNet controller needs an adapter. It is built through the
    # library factory so that parameter_space, tuning, metrics and stability
    # settings in the config are all applied.
    adapter = None
    if controller_cfg["type"] == "autotunenet":
        adapter_cfg = {
            "parameter_space": controller_cfg["parameter_space"],
            "tuning": controller_cfg.get("tuning", {}),
            "metrics": {"smoothing_window": 5, **controller_cfg.get("metrics", {})},
            "stability": controller_cfg.get("stability", {}),
        }
        adapter = build_pytorch_autotunenet(
            torch_optimizer=optimizer,
            raw_config=adapter_cfg,
            seed=seed,
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

    with open(output_dir / "run_info.json", "w") as f:
        json.dump(
            {
                "config": Path(args.config).name,
                "task": task,
                "seed": seed,
                "device": device.type,
                "torch_version": torch.__version__,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
