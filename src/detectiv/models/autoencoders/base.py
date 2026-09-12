from abc import ABC, abstractmethod

from torch import Tensor, nn


class ImageEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, images: Tensor) -> Tensor:
        raise NotImplementedError

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True


class ImageDecoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, embeddings: Tensor) -> Tensor:
        raise NotImplementedError


class ImageBottleneck(nn.Module, ABC):
    @abstractmethod
    def forward(self, embeddings: Tensor) -> Tensor:
        raise NotImplementedError
