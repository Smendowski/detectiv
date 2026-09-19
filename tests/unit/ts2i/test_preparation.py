import numpy as np

from detectiv.time_series import (
    TemporalBoundary,
    TemporalSplitter,
    TimeSeries,
    TimeSeriesDataset,
)
from detectiv.time_series.windowing import WindowSpec
from detectiv.ts2i import ImagePreparation
from detectiv.ts2i.channelization import MeanStdMaxChannelization
from detectiv.ts2i.projection import ConfiguredProjectionStrategy, ProjectionScheme
from detectiv.ts2i.transformations import Spiral


def test_preparation_allows_a_split_with_no_windows() -> None:
    dataset = TimeSeriesDataset(
        "series",
        {"series": TimeSeries(np.arange(6), series_id="series")},
    )
    projection = ConfiguredProjectionStrategy(
        ProjectionScheme(MeanStdMaxChannelization()).channels(
            Spiral(), Spiral(), Spiral()
        )
    )

    images = (
        ImagePreparation(dataset)
        .split(TemporalSplitter({"series": TemporalBoundary(train_end=3)}))
        .window(train=WindowSpec(4), test=WindowSpec(4))
        .project(projection)
        .build((4, 4))
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
    projection = ConfiguredProjectionStrategy(
        ProjectionScheme(MeanStdMaxChannelization()).channels(
            Spiral(), Spiral(), Spiral()
        )
    )

    inspection = (
        ImagePreparation(dataset)
        .split(TemporalSplitter({"series": TemporalBoundary(train_end=4)}))
        .window(train=WindowSpec(2), test=WindowSpec(2, stride=1))
        .project(projection)
        .inspect((4, 4))
    )

    assert inspection.train.image_count == 2
    assert inspection.test.image_shape.shape == (3, 4, 4)
    assert inspection.test.series["series"].windows == 3
    assert inspection.test.series["series"].covered_points == 4
    assert inspection.test.series["series"].uncovered_points == 0
    assert inspection.test.series["series"].maximum_coverage == 2
