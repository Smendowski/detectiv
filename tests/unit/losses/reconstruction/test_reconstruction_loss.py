import torch

from detectiv.losses import MeanSquaredReconstructionLoss


def test_mean_squared_reconstruction_loss_returns_scalar_mean() -> None:
    loss = MeanSquaredReconstructionLoss()

    value = loss(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 1.0]))

    assert value.item() == 2.5


def test_mean_squared_reconstruction_loss_exposes_pixel_errors() -> None:
    loss = MeanSquaredReconstructionLoss()

    error = loss.error(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 1.0]))

    assert torch.equal(error, torch.tensor([1.0, 4.0]))
