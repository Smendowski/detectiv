import torch.nn.functional as F
from torch import Tensor, nn


class MeanSquaredReconstructionLoss(nn.Module):
    """Mean squared error with elementwise reconstruction errors."""

    def forward(self, reconstruction: Tensor, images: Tensor) -> Tensor:
        """Return the mean squared error across every input element."""
        _validate_shapes(reconstruction, images)
        return F.mse_loss(reconstruction, images)

    def error(self, reconstruction: Tensor, images: Tensor) -> Tensor:
        """Return the elementwise squared reconstruction error."""
        _validate_shapes(reconstruction, images)
        return (images - reconstruction).square()


def _validate_shapes(reconstruction: Tensor, images: Tensor) -> None:
    if reconstruction.shape != images.shape:
        raise ValueError("reconstruction and images must have identical shapes")
