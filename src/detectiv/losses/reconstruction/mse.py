import torch.nn.functional as F
from torch import Tensor, nn


class MeanSquaredReconstructionLoss(nn.Module):
    def forward(self, reconstruction: Tensor, images: Tensor) -> Tensor:
        return F.mse_loss(reconstruction, images)

    def error(self, reconstruction: Tensor, images: Tensor) -> Tensor:
        return (images - reconstruction).square()
