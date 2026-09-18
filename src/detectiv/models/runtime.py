import torch


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if resolved.type == "cuda" and (
        resolved.index is not None and resolved.index >= torch.cuda.device_count()
    ):
        raise ValueError(f"CUDA device index is unavailable: {resolved.index}")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available")
    if resolved.type == "mps" and resolved.index not in (None, 0):
        raise ValueError(f"MPS device index is unavailable: {resolved.index}")
    return resolved
