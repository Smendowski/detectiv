import csv
import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from detectiv.data import TemporalSplit, WindowReference
from detectiv.datasets.images import ImageDataset, ImageShape, ImageSource


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
            self._write_metadata(temporary, images)
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

    def _write_metadata(
        self,
        root: Path,
        images: TemporalSplit[ImageDataset],
    ) -> None:
        datasets = [images.train, images.test]
        if images.validation is not None:
            datasets.append(images.validation)
        image_shape = images.train.image_shape
        if any(dataset.image_shape != image_shape for dataset in datasets):
            raise ValueError("all image splits must have the same image shape")
        dataset_ids = {"train": images.train.dataset_id, "test": images.test.dataset_id}
        series_lengths = {
            "train": dict(images.train.series_lengths),
            "test": dict(images.test.series_lengths),
        }
        if images.validation is not None:
            dataset_ids["validation"] = images.validation.dataset_id
            series_lengths["validation"] = dict(images.validation.series_lengths)
        metadata = {
            "format": self.config.format,
            "image_shape": {
                "channels": image_shape.channels,
                "height": image_shape.height,
                "width": image_shape.width,
            },
            "dataset_ids": dataset_ids,
            "series_lengths": series_lengths,
        }
        (root / "artifact.json").write_text(json.dumps(metadata), encoding="utf-8")

    def _write_split(
        self,
        root: Path,
        split: str,
        images: ImageDataset,
        writer: csv.DictWriter[str],
    ) -> None:
        for index in range(len(images)):
            window_label = None
            if images.window_labels is not None:
                window_label = bool(images.window_labels[index])
            label = ImageArtifactLabel.from_window_label(window_label)
            relative_path = Path(split) / label / (f"{index:06d}.{self.config.format}")
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            self._write_image(path, images[index], images.image_shape.channels)
            reference = images.window_references[index]
            writer.writerow(
                {
                    "split": split,
                    "label": label,
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


class ImageFolderSource(ImageSource):
    def __init__(self, paths: tuple[Path, ...], image_format: ImageFormat) -> None:
        self.paths = paths
        self.image_format = image_format

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> np.ndarray:
        path = self.paths[index]
        if self.image_format is ImageFormat.NPY:
            return np.asarray(np.load(path, allow_pickle=False))
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        return np.moveaxis(pixels, -1, 0)


class ImageFolderReader:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> TemporalSplit[ImageDataset]:
        metadata = self._read_metadata()
        image_format = self._image_format(metadata["format"])
        image_shape = self._image_shape(metadata["image_shape"])
        dataset_ids = metadata["dataset_ids"]
        if not isinstance(dataset_ids, dict):
            raise ValueError("artifact metadata must define dataset IDs")
        entries = self._read_manifest()
        datasets = {
            split: self._dataset(
                split,
                rows,
                dataset_ids,
                image_shape,
                image_format,
                self._series_lengths(metadata.get("series_lengths"), split),
            )
            for split, rows in entries.items()
            if rows or split in dataset_ids
        }
        if "train" not in datasets or "test" not in datasets:
            raise ValueError("artifact must define train and test splits")
        return TemporalSplit(
            train=datasets["train"],
            validation=datasets.get("validation"),
            test=datasets["test"],
        )

    def _read_metadata(self) -> dict[str, object]:
        path = self.path / "artifact.json"
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise FileNotFoundError(f"missing artifact metadata: {path}") from error
        if not isinstance(metadata, dict):
            raise ValueError("artifact metadata must be an object")
        return metadata

    @staticmethod
    def _image_format(value: object) -> ImageFormat:
        if not isinstance(value, str):
            raise ValueError("artifact metadata must define an image format")
        return ImageFormat(value)

    @staticmethod
    def _image_shape(value: object) -> ImageShape:
        if not isinstance(value, dict):
            raise ValueError("artifact metadata must define an image shape")
        dimensions = ("channels", "height", "width")
        if any(not isinstance(value.get(dimension), int) for dimension in dimensions):
            raise ValueError("artifact image dimensions must be integers")
        return ImageShape(
            channels=value["channels"],
            height=value["height"],
            width=value["width"],
        )

    def _read_manifest(self) -> dict[str, list[dict[str, str]]]:
        entries: dict[str, list[dict[str, str]]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        with (self.path / "manifest.csv").open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                split = row.get("split")
                if split not in entries:
                    raise ValueError(f"unknown artifact split: {split}")
                entries[split].append(row)
        return entries

    def _dataset(
        self,
        split: str,
        rows: list[dict[str, str]],
        dataset_ids: dict[object, object],
        image_shape: ImageShape,
        image_format: ImageFormat,
        series_lengths: dict[str, int] | None,
    ) -> ImageDataset:
        dataset_id = dataset_ids.get(split)
        if not isinstance(dataset_id, str):
            raise ValueError(f"artifact metadata has no dataset ID for {split}")
        paths: list[Path] = []
        references: list[WindowReference] = []
        labels: list[ImageArtifactLabel] = []
        for row in rows:
            paths.append(self._resolve(row["path"]))
            references.append(
                WindowReference(
                    row["series_id"],
                    int(row["start"]),
                    int(row["stop"]),
                    int(row["valid_length"]),
                )
            )
            labels.append(ImageArtifactLabel(row["label"]))
        if ImageArtifactLabel.UNLABELED in labels and len(set(labels)) > 1:
            raise ValueError("artifact split must be either labeled or unlabeled")
        window_labels = None
        if labels and labels[0] is not ImageArtifactLabel.UNLABELED:
            window_labels = np.asarray(
                [label.window_label for label in labels],
                dtype=bool,
            )
        return ImageDataset(
            dataset_id,
            image_shape=image_shape,
            window_references=references,
            source=ImageFolderSource(tuple(paths), image_format),
            window_labels=window_labels,
            series_lengths=series_lengths,
        )

    @staticmethod
    def _series_lengths(value: object, split: str) -> dict[str, int] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("artifact metadata must define series lengths by split")
        lengths = value.get(split)
        if not isinstance(lengths, dict):
            raise ValueError(f"artifact metadata has no series lengths for {split}")
        if any(
            not isinstance(series_id, str) or not isinstance(length, int)
            for series_id, length in lengths.items()
        ):
            raise ValueError("artifact series lengths must map IDs to integers")
        return lengths

    def _resolve(self, relative_path: str) -> Path:
        root = self.path.resolve()
        path = (root / relative_path).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"artifact path escapes its root: {relative_path}")
        return path
