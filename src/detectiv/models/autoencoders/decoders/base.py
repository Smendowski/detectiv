from abc import ABC, abstractmethod

from torch import Tensor, nn


class ImageDecoder(nn.Module, ABC):
    """Decode latent embeddings into channel-first images."""

    @abstractmethod
    def forward(self, embeddings: Tensor) -> Tensor:
        raise NotImplementedError
