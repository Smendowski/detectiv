import csv
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from detectiv.images import ImageDataset
from detectiv.images.io import ImageArtifactLabel, ImageFormat, ImageOutputConfig
from detectiv.time_series import TemporalSplit

IMAGE_ARTIFACT_MANIFEST_FIELDS = (
    "split",
    "label",
    "path",
    "series_id",
    "start",
    "stop",
    "valid_length",
)

POINT_LABELS_DIRECTORY = "point_labels"


def write_image_artifact_metadata(
    root: Path, images: TemporalSplit[ImageDataset], image_format: ImageFormat
) -> None:
    """Write image metadata and non-pickled point-label sidecars.

    Args:
        root: Existing artifact staging directory.
        images: Temporal image split to describe.
        image_format: Storage format used for image arrays.

    Raises:
        ValueError: If image splits are incompatible or metadata is not JSON
            serializable.
    """
    datasets = [images.train, images.test]
    if images.validation is not None:
        datasets.append(images.validation)

    image_shape = images.train.image_shape
    if any(dataset.image_shape != image_shape for dataset in datasets):
        raise ValueError("all image splits must have the same image shape")
    if image_format is ImageFormat.PNG and image_shape.channels != 3:
        raise ValueError("PNG output requires three image channels")

    dataset_ids = {"train": images.train.dataset_id, "test": images.test.dataset_id}
    series_lengths = {
        "train": dict(images.train.series_lengths),
        "test": dict(images.test.series_lengths),
    }
    dataset_metadata = {
        "train": dict(images.train.metadata),
        "test": dict(images.test.metadata),
    }
    if images.validation is not None:
        dataset_ids["validation"] = images.validation.dataset_id
        series_lengths["validation"] = dict(images.validation.series_lengths)
        dataset_metadata["validation"] = dict(images.validation.metadata)

    point_labels = _write_point_label_sidecars(root, images)

    metadata = {
        "format": image_format,
        "image_shape": {
            "channels": image_shape.channels,
            "height": image_shape.height,
            "width": image_shape.width,
        },
        "dataset_ids": dataset_ids,
        "series_lengths": series_lengths,
        "dataset_metadata": dataset_metadata,
    }
    if point_labels:
        metadata["point_labels"] = point_labels

    try:
        serialized = json.dumps(metadata)
    except TypeError as error:
        raise ValueError("image dataset metadata must be JSON serializable") from error

    (root / "artifact.json").write_text(serialized, encoding="utf-8")


def _write_point_label_sidecars(
    root: Path, images: TemporalSplit[ImageDataset]
) -> dict[str, dict[str, str]]:
    datasets = [("train", images.train), ("test", images.test)]
    if images.validation is not None:
        datasets.append(("validation", images.validation))

    references: dict[str, dict[str, str]] = {}
    for split, dataset in datasets:
        if dataset.point_labels is None:
            continue
        split_references: dict[str, str] = {}
        for index, (series_id, labels) in enumerate(dataset.point_labels.items()):
            relative_path = Path(POINT_LABELS_DIRECTORY) / split / f"{index:06d}.npy"
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, labels, allow_pickle=False)
            split_references[series_id] = relative_path.as_posix()
        references[split] = split_references
    return references


def image_artifact_row(
    split: str, images: ImageDataset, index: int, image_format: ImageFormat
) -> dict[str, str | int | ImageArtifactLabel]:
    """Build one manifest row for an image window.

    Args:
        split: Temporal split name.
        images: Source image dataset.
        index: Image index within the split.
        image_format: Artifact image format.

    Returns:
        Manifest fields describing the image and source window.
    """
    window_label = None
    if images.window_labels is not None:
        window_label = bool(images.window_labels[index])
    label = ImageArtifactLabel.from_window_label(window_label)
    relative_path = image_artifact_path(split, label, index, image_format)
    reference = images.window_references[index]
    return {
        "split": split,
        "label": label,
        "path": relative_path.as_posix(),
        "series_id": reference.series_id,
        "start": reference.start,
        "stop": reference.stop,
        "valid_length": reference.valid_length,
    }


def image_artifact_path(
    split: str, label: ImageArtifactLabel, index: int, image_format: ImageFormat
) -> Path:
    """Build an image's relative artifact path.

    Args:
        split: Temporal split name.
        label: Artifact label directory.
        index: Image index within the split.
        image_format: Artifact image format.

    Returns:
        Relative path for the image file.
    """
    return Path(split) / label / f"{index:06d}.{image_format}"


class ImageFolderWriter:
    """Write a temporal image split as a portable folder artifact."""

    def __init__(self, config: ImageOutputConfig) -> None:
        self.config = config

    def write(self, images: TemporalSplit[ImageDataset]) -> Path:
        """Write image splits and point-label sidecars to a folder artifact.

        Args:
            images: Temporal image split to serialize.

        Returns:
            Path to the completed folder artifact.

        Raises:
            FileExistsError: If the destination already exists.
        """
        output = self.config.path
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as name:
            temporary = Path(name)
            write_image_artifact_metadata(temporary, images, self.config.format)
            with (temporary / "manifest.csv").open("w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=IMAGE_ARTIFACT_MANIFEST_FIELDS,
                )
                writer.writeheader()
                self._write_split(temporary, "train", images.train, writer)
                if images.validation is not None:
                    self._write_split(
                        temporary, "validation", images.validation, writer
                    )
                self._write_split(temporary, "test", images.test, writer)
            os.replace(temporary, output)
        return output

    def _write_split(
        self, root: Path, split: str, images: ImageDataset, writer: csv.DictWriter[str]
    ) -> None:
        for index in range(len(images)):
            row = image_artifact_row(split, images, index, self.config.format)
            relative_path = Path(str(row["path"]))
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            self._write_image(path, images[index], images.image_shape.channels)
            writer.writerow(row)

    def _write_image(self, path: Path, image: np.ndarray, channels: int) -> None:
        if self.config.format is ImageFormat.NPY:
            if image.dtype.hasobject:
                raise ValueError("NPY output does not support object image arrays")
            np.save(path, image)
            return
        if channels != 3:
            raise ValueError("PNG output requires three image channels")
        if not np.isfinite(image).all() or (image < 0).any() or (image > 1).any():
            raise ValueError(
                "PNG output requires finite image values in the range [0, 1]"
            )
        pixels = (np.moveaxis(image, 0, -1) * 255).astype(np.uint8)
        Image.fromarray(pixels).save(path)
