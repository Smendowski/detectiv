import torch.nn.functional as F
from torch import Tensor, nn


class MeanSquaredReconstructionLoss(nn.Module):
    def forward(self, reconstruction: Tensor, images: Tensor) -> Tensor:
        _validate_shapes(reconstruction, images)
        return F.mse_loss(reconstruction, images)

    def error(self, reconstruction: Tensor, images: Tensor) -> Tensor:
        _validate_shapes(reconstruction, images)
        return (images - reconstruction).square()


def _validate_shapes(reconstruction: Tensor, images: Tensor) -> None:
    if reconstruction.shape != images.shape:
        raise ValueError("reconstruction and images must have identical shapes")
