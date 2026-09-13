from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ImageFormat(StrEnum):
    PNG = "png"
    NPY = "npy"


class ImageArtifactLabel(StrEnum):
    NORMAL = "0"
    ANOMALOUS = "1"
    UNLABELED = "unlabeled"

    @classmethod
    def from_window_label(cls, label: bool | None) -> "ImageArtifactLabel":
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
    path: Path
    format: ImageFormat = ImageFormat.NPY

    def __post_init__(self) -> None:
        if not isinstance(self.format, ImageFormat):
            raise TypeError("format must be an ImageFormat")
