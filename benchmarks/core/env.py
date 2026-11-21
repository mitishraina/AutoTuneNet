"""Environment details recorded with every result (comparisons need matching hardware)."""
import platform
import subprocess

import torch


def describe_environment(device: torch.device) -> dict:
    if device.type == "cuda":
        accelerator = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        accelerator = f"Apple MPS ({platform.machine()})"
    else:
        accelerator = f"CPU ({platform.processor() or platform.machine()})"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        commit = None
    return {
        "device": device.type,
        "accelerator": accelerator,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": commit,
    }
