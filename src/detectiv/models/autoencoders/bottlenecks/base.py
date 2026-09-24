from abc import ABC, abstractmethod

from torch import Tensor, nn


class ImageBottleneck(nn.Module, ABC):
    """Transform autoencoder embeddings between encoder and decoder."""

    @abstractmethod
    def forward(self, embeddings: Tensor) -> Tensor:
        raise NotImplementedError
