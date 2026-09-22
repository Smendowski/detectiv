import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from detectiv.images import ImageDataset, ImageShape, ImageSource
from detectiv.images.io import ImageArtifactLabel, ImageFormat
from detectiv.images.io.writers.images import IMAGE_ARTIFACT_SCHEMA_VERSION
from detectiv.time_series import TemporalSplit
from detectiv.time_series.windowing import WindowReference


class ImageFolderSource(ImageSource):
    """Lazy image source backed by ordered artifact file paths."""

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
    """Restore a temporal image split from a folder artifact."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> TemporalSplit[ImageDataset]:
        """Read and validate the folder metadata, sidecars, and manifest.

        Returns:
            Lazy train, optional validation, and test image datasets.

        Raises:
            FileNotFoundError: If required artifact metadata or manifest is absent.
            ValueError: If metadata, sidecars, or manifest rows are invalid.
        """
        metadata = self._read_metadata()
        if metadata.get("schema_version") != IMAGE_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported image artifact schema")
        image_format = self._image_format(metadata.get("format"))
        image_shape = self._image_shape(metadata.get("image_shape"))
        if image_format is ImageFormat.PNG and image_shape.channels != 3:
            raise ValueError("PNG artifacts must define three image channels")

        dataset_ids = metadata.get("dataset_ids")
        if not isinstance(dataset_ids, dict):
            raise ValueError("artifact metadata must define dataset IDs")
        dataset_metadata = metadata.get("dataset_metadata")
        if not isinstance(dataset_metadata, dict):
            raise ValueError("artifact metadata must define dataset metadata by split")

        entries = self._read_manifest()
        datasets = {
            split: self._dataset(
                split,
                rows,
                dataset_ids,
                image_shape,
                image_format,
                self._series(metadata.get("series"), split),
                self._point_labels(metadata.get("point_labels"), split),
                self._dataset_metadata(dataset_metadata, split),
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
        try:
            return ImageFormat(value)
        except ValueError as error:
            raise ValueError(f"unsupported artifact image format: {value}") from error

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
        required = {
            "split",
            "label",
            "path",
            "series_id",
            "start",
            "stop",
            "valid_length",
        }
        path = self.path / "manifest.csv"
        try:
            file = path.open(newline="", encoding="utf-8")
        except FileNotFoundError as error:
            raise FileNotFoundError(f"missing artifact manifest: {path}") from error
        with file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("artifact manifest is missing required columns")
            for row in reader:
                if any(row.get(field) is None for field in required):
                    raise ValueError("artifact manifest contains an incomplete row")
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
        series: tuple[str, int],
        point_labels: np.ndarray | None,
        metadata: dict[str, object],
    ) -> ImageDataset:
        dataset_id = dataset_ids.get(split)
        if not isinstance(dataset_id, str):
            raise ValueError(f"artifact metadata has no dataset ID for {split}")
        paths = [self._resolve(row["path"]) for row in rows]
        references: list[WindowReference] = []
        labels: list[ImageArtifactLabel] = []
        try:
            for row in rows:
                references.append(
                    WindowReference(
                        row["series_id"],
                        int(row["start"]),
                        int(row["stop"]),
                        int(row["valid_length"]),
                    )
                )
                labels.append(ImageArtifactLabel(row["label"]))
        except (KeyError, ValueError) as error:
            raise ValueError(f"artifact manifest has invalid {split} rows") from error
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
            series_id=series[0],
            series_length=series[1],
            window_labels=window_labels,
            point_labels=point_labels,
            metadata=metadata,
        )

    @staticmethod
    def _series(value: object, split: str) -> tuple[str, int]:
        if not isinstance(value, dict):
            raise ValueError("artifact metadata must define series by split")
        series = value.get(split)
        if not isinstance(series, dict):
            raise ValueError(f"artifact metadata has no series for {split}")
        series_id = series.get("id")
        length = series.get("length")
        if not isinstance(series_id, str) or not isinstance(length, int):
            raise ValueError(
                "artifact series must define a string ID and integer length"
            )
        return series_id, length

    @staticmethod
    def _dataset_metadata(value: dict[object, object], split: str) -> dict[str, object]:
        metadata = value.get(split)
        if not isinstance(metadata, dict) or not all(
            isinstance(key, str) for key in metadata
        ):
            raise ValueError(
                f"artifact metadata has no valid dataset metadata for {split}"
            )
        return metadata

    def _point_labels(self, value: object, split: str) -> np.ndarray | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("artifact metadata point labels must be mapped by split")
        reference = value.get(split)
        if reference is None:
            return None
        if not isinstance(reference, str):
            raise ValueError(
                f"artifact metadata has invalid point-label references for {split}"
            )
        try:
            return np.asarray(np.load(self._resolve(reference), allow_pickle=False))
        except (OSError, ValueError) as error:
            raise ValueError(
                f"artifact has invalid point labels for {split}"
            ) from error

    def _resolve(self, relative_path: str) -> Path:
        root = self.path.resolve()
        path = (root / relative_path).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"artifact path escapes its root: {relative_path}")
        if not path.is_file():
            raise ValueError(f"artifact image is missing: {relative_path}")
        return path
