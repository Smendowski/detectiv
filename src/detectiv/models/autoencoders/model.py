from typing import cast

from torch import Tensor, nn
from torch.nn import functional as functional

from detectiv.models.autoencoders.base import (
    ImageBottleneck,
    ImageDecoder,
    ImageEncoder,
)


class Autoencoder(nn.Module):
    def __init__(
        self,
        encoder: ImageEncoder,
        decoder: ImageDecoder,
        *,
        bottleneck: ImageBottleneck | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = bottleneck

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError("images must have shape (batch, channels, height, width)")
        embeddings = self.encoder(images)
        if self.bottleneck is not None:
            embeddings = self.bottleneck(embeddings)
        reconstruction = cast(Tensor, self.decoder(embeddings))
        if reconstruction.ndim != 4:
            raise ValueError("decoder must return a four-dimensional image tensor")
        if reconstruction.shape[-2:] != images.shape[-2:]:
            reconstruction = functional.interpolate(
                reconstruction,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return reconstruction

    def reconstruct(self, images: Tensor) -> Tensor:
        return self.forward(images)
