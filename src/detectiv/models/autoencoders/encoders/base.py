from abc import ABC, abstractmethod

from torch import Tensor, nn


class ImageEncoder(nn.Module, ABC):
    """Encode a batch of channel-first images into latent embeddings."""

    @abstractmethod
    def forward(self, images: Tensor) -> Tensor:
        """Return embeddings for a batch of images."""
        raise NotImplementedError

    def freeze(self) -> None:
        """Disable gradient updates for every encoder parameter."""
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze(self) -> None:
        """Enable gradient updates for every encoder parameter."""
        for parameter in self.parameters():
            parameter.requires_grad = True
