from typing import Self

import numpy as np

from detectiv.images import ImageSize
from detectiv.time_series import (
    TemporalBoundary,
    TemporalHoldout,
    TimeSeries,
    TimeSeriesSplit,
)
from detectiv.time_series.preprocessing import TimeSeriesPreprocessor
from detectiv.time_series.windowing import WindowedTimeSeriesSplit, WindowSpec
from detectiv.ts2i import ProjectedImageStage
from detectiv.ts2i.channelization import MeanStdMaxChannelization
from detectiv.ts2i.projection import FixedProjectionStrategy, ProjectionScheme
from detectiv.ts2i.transformations import Spiral


class RecordingPreprocessor(TimeSeriesPreprocessor):
    def __init__(self, offset: int = 0) -> None:
        self.offset = offset
        self.fitted_length: int | None = None
        self.fitted_means: list[float] = []
        self.transformed_lengths: list[int] = []
        self.transformed_means: list[float] = []

    def fit(self, train: TimeSeries) -> Self:
        self.fitted_length = train.n_timesteps
        self.fitted_means.append(float(train.values.mean()))
        return self

    def transform(self, series: TimeSeries) -> TimeSeries:
        self.transformed_lengths.append(series.n_timesteps)
        self.transformed_means.append(float(series.values.mean()))
        return TimeSeries(
            series.values + self.offset,
            labels=series.labels,
            feature_names=series.feature_names,
            sampling_rate=series.sampling_rate,
            series_id=series.series_id,
            metadata=series.metadata,
        )


def test_preparation_allows_a_split_with_no_windows() -> None:
    series = TimeSeries(np.arange(6), series_id="series")
    projection = FixedProjectionStrategy(
        ProjectionScheme(MeanStdMaxChannelization()).channels(
            Spiral(), Spiral(), Spiral()
        )
    )

    images = (
        series.split(TemporalBoundary(train_end=3))
        .window(train=WindowSpec(4), test=WindowSpec(4))
        .project(projection)
        .build(ImageSize(height=4, width=4))
    )

    assert len(images.train) == 0
    assert len(images.test) == 0
    assert images.train.series_length == 3
    assert images.test.series_length == 3


def test_preparation_resolves_a_temporal_holdout() -> None:
    series = TimeSeries(np.arange(10), series_id="series")
    projection = FixedProjectionStrategy(
        ProjectionScheme(MeanStdMaxChannelization()).channels(
            Spiral(), Spiral(), Spiral()
        )
    )

    images = (
        series.split(TemporalHoldout(test_start=8, validation_fraction=0.25))
        .window(
            train=WindowSpec(2),
            validation=WindowSpec(2),
            test=WindowSpec(2),
        )
        .project(projection)
        .build(ImageSize(height=4, width=4))
    )

    assert images.train.series_length == 6
    assert images.validation is not None
    assert images.validation.series_length == 2
    assert images.test.series_length == 2


def test_preparation_inspection_reports_window_coverage() -> None:
    series = TimeSeries(np.arange(8), series_id="series")
    projection = FixedProjectionStrategy(
        ProjectionScheme(MeanStdMaxChannelization()).channels(
            Spiral(), Spiral(), Spiral()
        )
    )

    inspection = (
        series.split(TemporalBoundary(train_end=4))
        .window(train=WindowSpec(2), test=WindowSpec(2, stride=1))
        .project(projection)
        .inspect(ImageSize(height=4, width=4))
    )

    assert inspection.train.image_count == 2
    assert inspection.test.image_shape.shape == (3, 4, 4)
    assert inspection.test.series_id == "series"
    assert inspection.test.series.windows == 3
    assert inspection.test.series.covered_points == 4
    assert inspection.test.series.uncovered_points == 0
    assert inspection.test.series.maximum_coverage == 2


def test_preparation_retains_partition_local_point_labels() -> None:
    labels = np.array([False, True, False, True, True, False])
    series = TimeSeries(np.arange(6), labels=labels, series_id="series")
    projection = FixedProjectionStrategy(
        ProjectionScheme(MeanStdMaxChannelization()).channels(
            Spiral(), Spiral(), Spiral()
        )
    )

    images = (
        series.split(TemporalBoundary(train_end=3))
        .window(train=WindowSpec(2), test=WindowSpec(2))
        .project(projection)
        .build(ImageSize(height=4, width=4))
    )

    assert images.train.point_labels is not None
    assert images.test.point_labels is not None
    np.testing.assert_array_equal(images.train.point_labels, labels[:3])
    np.testing.assert_array_equal(images.test.point_labels, labels[3:])


def test_preparation_preserves_unlabeled_partitions() -> None:
    series = TimeSeries(np.arange(6), series_id="series")
    projection = FixedProjectionStrategy(
        ProjectionScheme(MeanStdMaxChannelization()).channels(
            Spiral(), Spiral(), Spiral()
        )
    )

    images = (
        series.split(TemporalBoundary(train_end=3))
        .window(train=WindowSpec(2), test=WindowSpec(2))
        .project(projection)
        .build(ImageSize(height=4, width=4))
    )

    assert images.train.point_labels is None
    assert images.test.point_labels is None


def test_fluent_stages_fit_preprocessing_on_train_before_transforming_splits() -> None:
    preprocessor = RecordingPreprocessor()
    projection = FixedProjectionStrategy(
        ProjectionScheme(MeanStdMaxChannelization()).channels(
            Spiral(), Spiral(), Spiral()
        )
    )

    split = TimeSeries(np.arange(8), series_id="series").split(TemporalBoundary(4))
    windowed = split.preprocess(preprocessor).window(
        train=WindowSpec(2), test=WindowSpec(2)
    )
    projected = windowed.project(projection)
    projected.build(ImageSize(height=4, width=4))

    assert isinstance(split, TimeSeriesSplit)
    assert isinstance(windowed, WindowedTimeSeriesSplit)
    assert isinstance(projected, ProjectedImageStage)
    assert preprocessor.fitted_length == 4
    assert preprocessor.fitted_means == [1.5]
    assert preprocessor.transformed_lengths == [4, 4]
    assert preprocessor.transformed_means == [1.5, 5.5]


def test_repeated_preprocessing_composes_in_order_without_mutating_prior_stages() -> (
    None
):
    first = RecordingPreprocessor(offset=10)
    second = RecordingPreprocessor(offset=100)
    projection = FixedProjectionStrategy(
        ProjectionScheme(MeanStdMaxChannelization()).channels(
            Spiral(), Spiral(), Spiral()
        )
    )
    source = TimeSeries(np.arange(10), series_id="series").split(
        TemporalBoundary(train_end=4, validation_end=6)
    )

    first_stage = source.preprocess(first)
    second_stage = first_stage.preprocess(second)
    projected = second_stage.window(
        train=WindowSpec(2),
        validation=WindowSpec(2),
        test=WindowSpec(2),
    ).project(projection)
    projected.build(ImageSize(height=4, width=4))

    assert source._preprocessors == ()
    assert first_stage._preprocessors == (first,)
    assert second_stage._preprocessors == (first, second)
    np.testing.assert_array_equal(source.train.values.ravel(), [0, 1, 2, 3])
    assert first.fitted_means == [1.5]
    assert first.transformed_means == [1.5, 4.5, 7.5]
    assert second.fitted_means == [11.5]
    assert second.transformed_means == [11.5, 14.5, 17.5]
