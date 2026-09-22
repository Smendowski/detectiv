import numpy as np

from detectiv.images import ImageSize
from detectiv.time_series import (
    TemporalBoundary,
    TemporalSplitter,
    TimeSeries,
    TimeSeriesDataset,
)
from detectiv.time_series.windowing import WindowSpec
from detectiv.ts2i import ImagePreparation
from detectiv.ts2i.channelization import MeanStdMaxChannelization
from detectiv.ts2i.projection import FixedProjectionStrategy, ProjectionScheme
from detectiv.ts2i.transformations import Spiral


def test_preparation_allows_a_split_with_no_windows() -> None:
    dataset = TimeSeriesDataset(
        "series",
        {"series": TimeSeries(np.arange(6), series_id="series")},
    )
    projection = FixedProjectionStrategy(
        ProjectionScheme(MeanStdMaxChannelization()).channels(
            Spiral(), Spiral(), Spiral()
        )
    )

    images = (
        ImagePreparation(dataset)
        .split(TemporalSplitter({"series": TemporalBoundary(train_end=3)}))
        .window(train=WindowSpec(4), test=WindowSpec(4))
        .project(projection)
        .build(ImageSize(height=4, width=4))
    )

    assert len(images.train) == 0
    assert len(images.test) == 0
    assert images.train.series_lengths == {"series": 3}
    assert images.test.series_lengths == {"series": 3}


def test_preparation_inspection_reports_window_coverage() -> None:
    dataset = TimeSeriesDataset(
        "series",
        {"series": TimeSeries(np.arange(8), series_id="series")},
    )
    projection = FixedProjectionStrategy(
        ProjectionScheme(MeanStdMaxChannelization()).channels(
            Spiral(), Spiral(), Spiral()
        )
    )

    inspection = (
        ImagePreparation(dataset)
        .split(TemporalSplitter({"series": TemporalBoundary(train_end=4)}))
        .window(train=WindowSpec(2), test=WindowSpec(2, stride=1))
        .project(projection)
        .inspect(ImageSize(height=4, width=4))
    )

    assert inspection.train.image_count == 2
    assert inspection.test.image_shape.shape == (3, 4, 4)
    assert inspection.test.series["series"].windows == 3
    assert inspection.test.series["series"].covered_points == 4
    assert inspection.test.series["series"].uncovered_points == 0
    assert inspection.test.series["series"].maximum_coverage == 2


def test_preparation_retains_partition_local_point_labels() -> None:
    labels = np.array([False, True, False, True, True, False])
    dataset = TimeSeriesDataset(
        "series",
        {"series": TimeSeries(np.arange(6), labels=labels, series_id="series")},
    )
    projection = FixedProjectionStrategy(
        ProjectionScheme(MeanStdMaxChannelization()).channels(
            Spiral(), Spiral(), Spiral()
        )
    )

    images = (
        ImagePreparation(dataset)
        .split(TemporalSplitter({"series": TemporalBoundary(train_end=3)}))
        .window(train=WindowSpec(2), test=WindowSpec(2))
        .project(projection)
        .build(ImageSize(height=4, width=4))
    )

    assert images.train.point_labels is not None
    assert images.test.point_labels is not None
    np.testing.assert_array_equal(images.train.point_labels["series"], labels[:3])
    np.testing.assert_array_equal(images.test.point_labels["series"], labels[3:])


def test_preparation_preserves_unlabeled_partitions() -> None:
    dataset = TimeSeriesDataset(
        "series",
        {"series": TimeSeries(np.arange(6), series_id="series")},
    )
    projection = FixedProjectionStrategy(
        ProjectionScheme(MeanStdMaxChannelization()).channels(
            Spiral(), Spiral(), Spiral()
        )
    )

    images = (
        ImagePreparation(dataset)
        .split(TemporalSplitter({"series": TemporalBoundary(train_end=3)}))
        .window(train=WindowSpec(2), test=WindowSpec(2))
        .project(projection)
        .build(ImageSize(height=4, width=4))
    )

    assert images.train.point_labels is None
    assert images.test.point_labels is None
