from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum

import torch
from torch import nn


class ComputeDevice(StrEnum):
    """Common compute-device selections for model training and inference."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@contextmanager
def evaluating(model: nn.Module) -> Iterator[None]:
    """Temporarily evaluate a model without losing mixed child-module modes."""
    modes = tuple((module, module.training) for module in model.modules())
    model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


def resolve_device(device: ComputeDevice | str = ComputeDevice.AUTO) -> torch.device:
    """Resolve and validate an explicit or automatically selected Torch device."""
    if device is ComputeDevice.AUTO or device == ComputeDevice.AUTO:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(str(device))
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
