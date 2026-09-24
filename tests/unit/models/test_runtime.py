import pytest
import torch

from detectiv.models import ComputeDevice
from detectiv.models.runtime import evaluating, resolve_device


def test_evaluating_restores_mixed_modes_after_an_error() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    model.train()
    model[0].eval()

    with pytest.raises(RuntimeError, match="failed"), evaluating(model):
        assert not any(module.training for module in model.modules())
        raise RuntimeError("failed")

    assert model.training
    assert not model[0].training
    assert model[1].training


def test_resolve_device_accepts_a_compute_device() -> None:
    assert resolve_device(ComputeDevice.CPU) == torch.device("cpu")


def test_resolve_device_rejects_an_unavailable_cuda_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(ValueError, match="index is unavailable"):
        resolve_device("cuda:1")
