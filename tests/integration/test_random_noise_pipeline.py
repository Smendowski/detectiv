from pathlib import Path

import numpy as np
import pytest

from detectiv import (
    FixedProjectionStrategy,
    Identity,
    ImageFolderWriter,
    ImageOutputConfig,
    ImagePreparation,
    ProjectionScheme,
    RandomNoise,
    TailPolicy,
    TemporalBoundary,
    TemporalSplitter,
    TimeSeries,
    TimeSeriesDataset,
    WindowSpec,
)


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
        ProjectionScheme(Identity()).channels(RandomNoise()).replicate(n_channels=3)
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
                ProjectionScheme(Identity())
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
