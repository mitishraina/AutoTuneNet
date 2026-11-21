import torch


def resolve_device(name: str = "auto") -> torch.device:
    """'auto' picks CUDA, then Apple MPS, then CPU."""
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
