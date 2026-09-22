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


class ImageDecoder(nn.Module, ABC):
    """Decode latent embeddings into channel-first images."""

    @abstractmethod
    def forward(self, embeddings: Tensor) -> Tensor:
        raise NotImplementedError


class ImageBottleneck(nn.Module, ABC):
    """Transform autoencoder embeddings between encoder and decoder."""

    @abstractmethod
    def forward(self, embeddings: Tensor) -> Tensor:
        raise NotImplementedError
