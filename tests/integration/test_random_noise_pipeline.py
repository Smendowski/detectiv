import zipfile
from pathlib import Path

import numpy as np
import pytest

from detectiv.data import TemporalSplit, TimeSeries, WindowReference
from detectiv.datasets import (
    ImageArchiveWriter,
    ImageArtifactReader,
    ImageDataset,
    ImageFolderReader,
    ImageFolderWriter,
    ImageFormat,
    ImageOutputConfig,
    ImageShape,
    ImageSource,
    TemporalBoundary,
    TemporalSplitter,
    TimeSeriesDataset,
)
from detectiv.ts2i import ImagePreparation
from detectiv.ts2i.channelization import IdentityChannelization
from detectiv.ts2i.channelization.context import WindowContext
from detectiv.ts2i.projection import FixedProjectionStrategy, ProjectionScheme
from detectiv.ts2i.transformations import RandomNoise
from detectiv.windowing import (
    TailPolicy,
    WindowSpec,
)


class ArrayImageSource(ImageSource):
    def __init__(self, images: list[np.ndarray]) -> None:
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.images[index]


def test_random_noise_pipeline_builds_lazy_reproducible_images() -> None:
    dataset = TimeSeriesDataset(
        "example",
        {
            "series": TimeSeries(
                np.arange(16).reshape(8, 2),
                labels=np.array([0, 0, 0, 0, 0, 0, 1, 0]),
                series_id="series",
            )
        },
    )
    projection = FixedProjectionStrategy(
        ProjectionScheme(IdentityChannelization(WindowContext()))
        .channels(RandomNoise())
        .replicate(n_channels=3)
    )

    images = (
        ImagePreparation(dataset)
        .split(TemporalSplitter({"series": TemporalBoundary(4)}))
        .window(
            train=WindowSpec(2),
            test=WindowSpec(2, stride=1, tail=TailPolicy.EDGE_PAD),
        )
        .project(projection)
        .build((3, 4), seed=7)
    )

    assert len(images.train) == 2
    assert len(images.test) == 4
    assert images.test.window_references[-1].valid_length == 1
    first = images.test[0]
    np.testing.assert_array_equal(first, images.test[0])
    np.testing.assert_array_equal(first[0], first[1])
    np.testing.assert_array_equal(first[1], first[2])
    assert images.train.window_labels is not None
    assert images.train.window_labels.tolist() == [False, False]
    assert images.test.window_labels is not None
    assert images.test.window_labels.tolist() == [False, True, True, False]


def test_image_folder_writer_preserves_window_order(tmp_path: Path) -> None:
    dataset = TimeSeriesDataset(
        "example",
        {
            "series": TimeSeries(
                np.arange(16).reshape(8, 2),
                labels=np.array([0, 0, 0, 0, 0, 0, 1, 0]),
                series_id="series",
            )
        },
    )
    images = (
        ImagePreparation(dataset)
        .split(TemporalSplitter({"series": TemporalBoundary(4)}))
        .window(train=WindowSpec(2), test=WindowSpec(2, stride=1))
        .project(
            FixedProjectionStrategy(
                ProjectionScheme(IdentityChannelization(WindowContext()))
                .channels(RandomNoise())
                .replicate(n_channels=3)
            )
        )
        .build((3, 4), seed=7)
    )

    output = ImageFolderWriter(ImageOutputConfig(tmp_path / "images")).write(images)

    assert (output / "train/0/000000.png").exists()
    assert (output / "test/1/000001.png").exists()
    manifest = (output / "manifest.csv").read_text().splitlines()
    assert manifest[1].split(",") == [
        "train",
        "0",
        "train/0/000000.png",
        "series",
        "0",
        "2",
        "2",
    ]
    assert manifest[-1].split(",") == [
        "test",
        "1",
        "test/1/000002.png",
        "series",
        "2",
        "4",
        "2",
    ]


def test_parallel_image_export_is_explicitly_not_implemented(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="parallel"):
        ImageOutputConfig(tmp_path / "images", workers=1)


def test_image_artifacts_round_trip_lazily(tmp_path: Path) -> None:
    dataset = TimeSeriesDataset(
        "example",
        {
            "series": TimeSeries(
                np.arange(16).reshape(8, 2),
                labels=np.array([0, 0, 0, 0, 0, 0, 1, 0]),
                series_id="series",
            )
        },
    )
    images = (
        ImagePreparation(dataset)
        .split(TemporalSplitter({"series": TemporalBoundary(4)}))
        .window(train=WindowSpec(2), test=WindowSpec(2, stride=1))
        .project(
            FixedProjectionStrategy(
                ProjectionScheme(IdentityChannelization(WindowContext()))
                .channels(RandomNoise())
                .replicate(n_channels=3)
            )
        )
        .build((3, 4), seed=7)
    )

    folder = ImageFolderWriter(
        ImageOutputConfig(tmp_path / "images", format=ImageFormat.NPY)
    ).write(images)
    restored = ImageFolderReader(folder).read()

    assert restored.test.source.__class__.__name__ == "ImageFolderSource"
    np.testing.assert_array_equal(restored.test[0], images.test[0])
    assert restored.test.window_references == images.test.window_references
    assert restored.test.series_lengths == images.test.series_lengths
    np.testing.assert_array_equal(
        restored.test.window_labels,
        images.test.window_labels,
    )

    archive = ImageArchiveWriter(tmp_path / "images.zip", ImageFormat.NPY).write(images)
    with ImageArtifactReader(archive).open() as extracted:
        np.testing.assert_array_equal(extracted.test[0], images.test[0])
        assert extracted.test.window_references == images.test.window_references


def test_image_archive_reader_rejects_unsafe_member_paths(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "x") as output:
        output.writestr("../outside", "unsafe")

    with (
        pytest.raises(ValueError, match="escapes"),
        ImageArtifactReader(archive).open(),
    ):
        pass


def test_image_folder_round_trips_unlabeled_images(tmp_path: Path) -> None:
    image = np.ones((3, 2, 2), dtype=np.float32)
    dataset = ImageDataset(
        "unlabeled",
        image_shape=ImageShape(3, 2, 2),
        window_references=[WindowReference("series", 0, 2, 2)],
        source=ArrayImageSource([image]),
    )
    split = TemporalSplit(train=dataset, test=dataset)

    folder = ImageFolderWriter(
        ImageOutputConfig(tmp_path / "unlabeled", format=ImageFormat.NPY)
    ).write(split)
    restored = ImageFolderReader(folder).read()

    assert restored.train.window_labels is None
    np.testing.assert_array_equal(restored.train[0], image)
