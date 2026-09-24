from typing import cast

from torch import Tensor, nn
from torch.nn import functional as functional

from detectiv.models.autoencoders.bottlenecks.base import BaseBottleneck
from detectiv.models.autoencoders.decoders.base import BaseDecoder
from detectiv.models.autoencoders.encoders.base import BaseEncoder


class Autoencoder(nn.Module):
    """Compose an image encoder, optional bottleneck, and image decoder."""

    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        *,
        bottleneck: BaseBottleneck | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = bottleneck

    def forward(self, images: Tensor) -> Tensor:
        """Reconstruct a batch of channel-first images."""
        if images.ndim != 4:
            raise ValueError("images must have shape (batch, channels, height, width)")
        embeddings = self.encoder(images)
        if self.bottleneck is not None:
            embeddings = self.bottleneck(embeddings)
        reconstruction = cast(Tensor, self.decoder(embeddings))
        if reconstruction.ndim != 4:
            raise ValueError("decoder must return a four-dimensional image tensor")
        if reconstruction.shape[:2] != images.shape[:2]:
            raise ValueError(
                "decoder must preserve the input batch size and channel count"
            )
        if reconstruction.shape[-2:] != images.shape[-2:]:
            reconstruction = functional.interpolate(
                reconstruction,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return reconstruction

    def reconstruct(self, images: Tensor) -> Tensor:
        """Return a reconstruction of a batch of images."""
        return self.forward(images)
