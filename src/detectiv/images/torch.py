from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from detectiv.images.dataset import ImageDataset


class TorchImageDataset(TorchDataset[Tensor]):
    """PyTorch adapter for all or an ordered subset of an image dataset."""

    def __init__(
        self,
        images: ImageDataset,
        indices: Sequence[int] | NDArray[np.intp] | None = None,
    ) -> None:
        self.images = images
        self._indices = np.arange(len(images), dtype=np.intp)
        if indices is not None:
            values = np.asarray(indices)
            if values.ndim != 1 or (
                len(values) and not np.issubdtype(values.dtype, np.integer)
            ):
                raise ValueError("indices must be a one-dimensional integer sequence")
            self._indices = np.array(values, dtype=np.intp, copy=True)
        if self._indices.ndim != 1 or (
            len(self._indices)
            and ((self._indices < 0).any() or (self._indices >= len(images)).any())
        ):
            raise ValueError("indices must select images from the dataset")
        self._indices.setflags(write=False)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> Tensor:
        image = np.ascontiguousarray(self.images[int(self._indices[index])])
        return torch.as_tensor(image, dtype=torch.float32)
