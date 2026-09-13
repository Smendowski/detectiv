import json
from pathlib import Path

import numpy as np
import pytest

from detectiv.images import ImageDataset, ImageShape, ImageSource
from detectiv.images.io import ImageFormat, ImageOutputConfig
from detectiv.images.io.readers import ImageFolderReader
from detectiv.images.io.writers import ImageArchiveWriter, ImageFolderWriter
from detectiv.time_series import TemporalSplit
from detectiv.time_series.windowing import WindowReference


class ArrayImageSource(ImageSource):
    def __init__(self, image: np.ndarray) -> None:
        self.image = image

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> np.ndarray:
        return self.image


def test_output_config_requires_an_image_format(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="ImageFormat"):
        ImageOutputConfig(tmp_path / "images", format="npy")  # type: ignore[arg-type]

    assert ImageOutputConfig(tmp_path / "images").format is ImageFormat.NPY


@pytest.mark.parametrize("image", [np.full((3, 2, 2), -0.1), np.full((3, 2, 2), 1.1)])
def test_png_output_rejects_values_outside_its_numeric_domain(
    tmp_path: Path, image: np.ndarray
) -> None:
    output = tmp_path / "images"

    with pytest.raises(ValueError, match="range"):
        ImageFolderWriter(ImageOutputConfig(output, format=ImageFormat.PNG)).write(
            _split(image)
        )

    assert not output.exists()


def test_archive_writer_requires_an_image_format(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="ImageFormat"):
        ImageArchiveWriter(tmp_path / "images.zip", image_format="npy")  # type: ignore[arg-type]


def test_npy_artifacts_preserve_json_safe_dataset_metadata(tmp_path: Path) -> None:
    metadata = {"source": "example", "seed": 7}
    folder = ImageFolderWriter(ImageOutputConfig(tmp_path / "images")).write(
        _split(np.ones((3, 2, 2)), metadata=metadata)
    )

    restored = ImageFolderReader(folder).read()

    assert restored.train.metadata == metadata
    assert restored.test.metadata == metadata


def test_writer_rejects_non_serializable_dataset_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JSON serializable"):
        ImageFolderWriter(ImageOutputConfig(tmp_path / "images")).write(
            _split(np.ones((3, 2, 2)), metadata={"path": tmp_path})
        )


def test_reader_rejects_png_artifacts_without_rgb_shape(tmp_path: Path) -> None:
    (tmp_path / "artifact.json").write_text(
        json.dumps(
            {
                "format": "png",
                "image_shape": {"channels": 1, "height": 2, "width": 2},
                "dataset_ids": {"train": "images", "test": "images"},
                "series_lengths": {"train": {}, "test": {}},
                "dataset_metadata": {"train": {}, "test": {}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "manifest.csv").write_text(
        "split,label,path,series_id,start,stop,valid_length\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="three image channels"):
        ImageFolderReader(tmp_path).read()


def _split(
    image: np.ndarray, *, metadata: dict[str, object] | None = None
) -> TemporalSplit[ImageDataset]:
    dataset = ImageDataset(
        "images",
        image_shape=ImageShape(3, 2, 2),
        window_references=(WindowReference("series", 0, 2, 2),),
        source=ArrayImageSource(image),
        metadata=metadata,
    )
    return TemporalSplit(train=dataset, test=dataset)
