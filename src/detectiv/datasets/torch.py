from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from detectiv.datasets.images import ImageDataset


class TorchImageDataset(TorchDataset[Tensor]):
    def __init__(
        self,
        images: ImageDataset,
        indices: Sequence[int] | NDArray[np.intp] | None = None,
    ) -> None:
        self.images = images
        self.indices = np.arange(len(images), dtype=np.intp)
        if indices is not None:
            self.indices = np.asarray(indices, dtype=np.intp)
        if self.indices.ndim != 1 or (
            len(self.indices)
            and ((self.indices < 0).any() or (self.indices >= len(images)).any())
        ):
            raise ValueError("indices must select images from the dataset")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tensor:
        image = np.ascontiguousarray(self.images[int(self.indices[index])])
        return torch.as_tensor(image, dtype=torch.float32)
