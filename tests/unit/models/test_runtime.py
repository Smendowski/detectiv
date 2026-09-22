import pytest
import torch

from detectiv.models import ComputeDevice
from detectiv.models.runtime import resolve_device


def test_resolve_device_accepts_a_compute_device() -> None:
    assert resolve_device(ComputeDevice.CPU) == torch.device("cpu")


def test_resolve_device_rejects_an_unavailable_cuda_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(ValueError, match="index is unavailable"):
        resolve_device("cuda:1")
