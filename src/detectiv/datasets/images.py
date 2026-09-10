from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from detectiv.data import WindowReference
from detectiv.datasets.base import Dataset


@dataclass(frozen=True)
class ImageShape:
    channels: int
    height: int
    width: int

    def __post_init__(self) -> None:
        if self.channels <= 0 or self.height <= 0 or self.width <= 0:
            raise ValueError("image dimensions must be positive")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.channels, self.height, self.width)


class ImageSource(ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> np.ndarray:
        raise NotImplementedError


class ImageDataset(Dataset[int, np.ndarray]):
    def __init__(
        self,
        dataset_id: str,
        *,
        image_shape: ImageShape,
        window_references: Sequence[WindowReference],
        source: ImageSource,
        window_labels: np.ndarray | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(dataset_id, metadata=metadata)
        if len(source) != len(window_references):
            raise ValueError("source and window_references must have the same length")
        self.image_shape = image_shape
        self.window_references = tuple(window_references)
        self.source = source
        self.window_labels: np.ndarray | None = None
        if window_labels is not None:
            labels = np.asarray(window_labels, dtype=bool)
            if labels.ndim != 1 or len(labels) != len(self):
                raise ValueError("window_labels must have one value per image")
            self.window_labels = np.array(labels, copy=True)
            self.window_labels.setflags(write=False)

    def __len__(self) -> int:
        return len(self.window_references)

    def __getitem__(self, index: int) -> np.ndarray:
        if not -len(self) <= index < len(self):
            raise IndexError("image index out of range")

        image = np.asarray(self.source[index])
        if image.shape != self.image_shape.shape:
            raise ValueError(
                "image provider returned "
                f"{image.shape}; expected {self.image_shape.shape}"
            )
        return image
