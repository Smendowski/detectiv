import csv
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from detectiv.data import TemporalSplit
from detectiv.datasets.images import ImageDataset


class ImageFormat(StrEnum):
    PNG = "png"
    NPY = "npy"


@dataclass(frozen=True)
class ImageOutputConfig:
    path: Path
    format: ImageFormat = ImageFormat.PNG
    workers: int = 0

    def __post_init__(self) -> None:
        if self.workers < 0:
            raise ValueError("workers must not be negative")
        if self.workers:
            raise NotImplementedError("parallel image export is not implemented")


class ImageFolderWriter:
    def __init__(self, config: ImageOutputConfig) -> None:
        self.config = config

    def write(self, images: TemporalSplit[ImageDataset]) -> Path:
        output = self.config.path
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as name:
            temporary = Path(name)
            with (temporary / "manifest.csv").open("w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=(
                        "split",
                        "label",
                        "path",
                        "series_id",
                        "start",
                        "stop",
                        "valid_length",
                    ),
                )
                writer.writeheader()
                self._write_split(temporary, "train", images.train, writer)
                if images.validation is not None:
                    self._write_split(
                        temporary,
                        "validation",
                        images.validation,
                        writer,
                    )
                self._write_split(temporary, "test", images.test, writer)
            os.replace(temporary, output)
        return output

    def _write_split(
        self,
        root: Path,
        split: str,
        images: ImageDataset,
        writer: csv.DictWriter[str],
    ) -> None:
        if images.window_labels is None:
            raise ValueError("image-folder output requires point-level labels")
        for index, label in enumerate(images.window_labels):
            relative_path = (
                Path(split) / str(int(label)) / (f"{index:06d}.{self.config.format}")
            )
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            self._write_image(path, images[index], images.image_shape.channels)
            reference = images.window_references[index]
            writer.writerow(
                {
                    "split": split,
                    "label": int(label),
                    "path": relative_path.as_posix(),
                    "series_id": reference.series_id,
                    "start": reference.start,
                    "stop": reference.stop,
                    "valid_length": reference.valid_length,
                }
            )

    def _write_image(self, path: Path, image: np.ndarray, channels: int) -> None:
        if self.config.format is ImageFormat.NPY:
            np.save(path, image)
            return
        if channels != 3:
            raise ValueError("PNG output requires three image channels")
        pixels = np.clip(np.moveaxis(image, 0, -1) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(pixels).save(path)
