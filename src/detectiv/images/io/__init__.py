from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ImageFormat(StrEnum):
    """Portable image artifact encodings."""

    PNG = "png"
    NPY = "npy"


class ImageArtifactLabel(StrEnum):
    """Serialized window-label values used by image artifacts."""

    NORMAL = "0"
    ANOMALOUS = "1"
    UNLABELED = "unlabeled"

    @classmethod
    def from_window_label(cls, label: bool | None) -> ImageArtifactLabel:
        if label is None:
            return cls.UNLABELED
        return cls.ANOMALOUS if label else cls.NORMAL

    @property
    def window_label(self) -> bool | None:
        if self is self.UNLABELED:
            return None
        return self is self.ANOMALOUS


@dataclass(frozen=True)
class ImageOutputConfig:
    """Destination and encoding for an image folder artifact."""

    path: Path
    format: ImageFormat = ImageFormat.NPY

    def __post_init__(self) -> None:
        if not isinstance(self.format, ImageFormat):
            raise TypeError("format must be an ImageFormat")
