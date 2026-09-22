import numpy as np
import pytest

from detectiv.time_series import TemporalBoundary, TimeSeries
from detectiv.time_series.windowing import (
    SplitPart,
    SplitWindowing,
    WindowedTimeSeriesSplit,
    WindowSpec,
)


class RecordingProjection:
    def __init__(self) -> None:
        self.source: WindowedTimeSeriesSplit | None = None

    def project(self, source: WindowedTimeSeriesSplit) -> str:
        self.source = source
        return "projected"


def test_split_windowing_returns_the_spec_for_each_partition() -> None:
    train = WindowSpec(64)
    validation = WindowSpec(32, stride=1)
    test = WindowSpec(16, stride=4)
    windowing = SplitWindowing(train=train, validation=validation, test=test)

    assert windowing.spec_for(SplitPart.TRAIN) is train
    assert windowing.spec_for(SplitPart.VALIDATION) is validation
    assert windowing.spec_for(SplitPart.TEST) is test
    assert windowing.spec_for("test") is test


def test_split_windowing_rejects_an_unconfigured_validation_spec() -> None:
    windowing = SplitWindowing(train=WindowSpec(64), test=WindowSpec(64))

    with pytest.raises(ValueError, match="not configured"):
        windowing.spec_for(SplitPart.VALIDATION)


def test_split_windowing_rejects_unknown_split_parts() -> None:
    windowing = SplitWindowing(train=WindowSpec(64), test=WindowSpec(64))

    with pytest.raises(ValueError, match="unsupported split part"):
        windowing.spec_for("evaluation")


def test_windowed_split_delegates_projection_without_owning_the_result_type() -> None:
    windowed = (
        TimeSeries(np.arange(8), series_id="series")
        .split(TemporalBoundary(4))
        .window(train=WindowSpec(2), test=WindowSpec(2))
    )
    projection = RecordingProjection()

    result = windowed.project(projection)

    assert result == "projected"
    assert projection.source is windowed
