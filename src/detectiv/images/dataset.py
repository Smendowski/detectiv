from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from detectiv.time_series.windowing.reference import WindowReference


@dataclass(frozen=True)
class ImageShape:
    """Channel-first shape shared by every image in a dataset."""

    channels: int
    height: int
    width: int

    def __post_init__(self) -> None:
        if self.channels <= 0 or self.height <= 0 or self.width <= 0:
            raise ValueError("image dimensions must be positive")

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the channel-first dimensions.

        Returns:
            Channel, height, and width dimensions.
        """
        return (self.channels, self.height, self.width)


class ImageSource(ABC):
    """Lazy indexed provider of channel-first image arrays."""

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> np.ndarray:
        raise NotImplementedError


class ImageDataset:
    """Lazy images, source windows, and partition-local series metadata.

    `point_labels` retains immutable labels for every source time point and is
    distinct from optional `window_labels` derived during windowing. Labelled
    datasets require one aligned point-label array for every `series_lengths`
    entry; unlabelled datasets preserve `None`.
    """

    def __init__(
        self,
        dataset_id: str,
        *,
        image_shape: ImageShape,
        window_references: Sequence[WindowReference],
        source: ImageSource,
        window_labels: np.ndarray | None = None,
        series_lengths: Mapping[str, int] | None = None,
        point_labels: Mapping[str, np.ndarray] | None = None,
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
        self.series_lengths = _validate_series_lengths(
            self.window_references, series_lengths
        )
        self.point_labels = _validate_point_labels(point_labels, self.series_lengths)
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


def _validate_series_lengths(
    references: tuple[WindowReference, ...],
    series_lengths: Mapping[str, int] | None,
) -> Mapping[str, int]:
    series_ids = {reference.series_id for reference in references}
    if series_lengths is None:
        lengths = {
            series_id: max(
                reference.stop
                for reference in references
                if reference.series_id == series_id
            )
            for series_id in series_ids
        }
    else:
        lengths = dict(series_lengths)
        if not series_ids <= set(lengths):
            raise ValueError("series_lengths must define every referenced series")
    if any(length <= 0 for length in lengths.values()):
        raise ValueError("series lengths must be positive")
    if any(reference.stop > lengths[reference.series_id] for reference in references):
        raise ValueError("window references must lie within their series length")
    return MappingProxyType(lengths)


def _validate_point_labels(
    point_labels: Mapping[str, np.ndarray] | None,
    series_lengths: Mapping[str, int],
) -> Mapping[str, np.ndarray] | None:
    if point_labels is None:
        return None
    if set(point_labels) != set(series_lengths):
        raise ValueError("point_labels must define every series and no others")

    validated: dict[str, np.ndarray] = {}
    for series_id, values in point_labels.items():
        labels = np.asarray(values)
        if labels.ndim != 1 or len(labels) != series_lengths[series_id]:
            raise ValueError(
                f"point_labels for {series_id!r} must match its series length"
            )
        if not np.issubdtype(labels.dtype, np.bool_) and not np.all(
            (labels == 0) | (labels == 1)
        ):
            raise ValueError("point_labels must contain only binary values")
        immutable = np.array(labels, dtype=bool, copy=True)
        immutable.setflags(write=False)
        validated[series_id] = immutable
    return MappingProxyType(validated)
