from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import numpy as np

from detectiv.images.types import ImageShape
from detectiv.time_series.windowing.reference import WindowReference


class ImageSource(ABC):
    """Lazy indexed provider of channel-first image arrays."""

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> np.ndarray:
        raise NotImplementedError


class ImageDataset:
    """Lazy images and source windows for one partition of one series.

    `point_labels` retains immutable labels for every source time point and is
    distinct from optional `window_labels` derived during windowing. Labelled
    labelled datasets require one aligned point-label array; unlabelled datasets
    preserve `None`.
    """

    def __init__(
        self,
        dataset_id: str,
        *,
        image_shape: ImageShape,
        window_references: Sequence[WindowReference],
        source: ImageSource,
        series_id: str,
        series_length: int,
        window_labels: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if not dataset_id:
            raise ValueError("dataset_id must not be empty")
        if len(source) != len(window_references):
            raise ValueError("source and window_references must have the same length")
        self.dataset_id = dataset_id
        self.metadata = MappingProxyType(dict(metadata or {}))
        self.image_shape = image_shape
        self.window_references = tuple(window_references)
        self.source = source
        if not series_id:
            raise ValueError("series_id must not be empty")
        if isinstance(series_length, bool) or not isinstance(series_length, int):
            raise ValueError("series_length must be a positive integer")
        if series_length <= 0:
            raise ValueError("series_length must be a positive integer")
        if any(
            reference.series_id != series_id for reference in self.window_references
        ):
            raise ValueError("all window references must belong to series_id")
        if any(reference.stop > series_length for reference in self.window_references):
            raise ValueError("window references must lie within series_length")
        self.series_id = series_id
        self.series_length = series_length
        self.point_labels = _validate_point_labels(point_labels, series_length)
        self.window_labels: np.ndarray | None = None
        if window_labels is not None:
            labels = np.asarray(window_labels)
            if labels.ndim != 1 or len(labels) != len(self):
                raise ValueError("window_labels must have one value per image")
            if not np.issubdtype(labels.dtype, np.bool_) and not np.all(
                (labels == 0) | (labels == 1)
            ):
                raise ValueError("window_labels must contain only binary values")
            self.window_labels = np.array(labels, dtype=bool, copy=True)
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


def _validate_point_labels(
    point_labels: np.ndarray | None,
    series_length: int,
) -> np.ndarray | None:
    if point_labels is None:
        return None
    labels = np.asarray(point_labels)
    if labels.ndim != 1 or len(labels) != series_length:
        raise ValueError("point_labels must match series_length")
    if not np.issubdtype(labels.dtype, np.bool_) and not np.all(
        (labels == 0) | (labels == 1)
    ):
        raise ValueError("point_labels must contain only binary values")
    immutable = np.array(labels, dtype=bool, copy=True)
    immutable.setflags(write=False)
    return immutable
