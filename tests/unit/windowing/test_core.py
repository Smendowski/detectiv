import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from detectiv import (
    TailPolicy,
    TimeSeries,
    Windower,
    WindowLabelingStrategy,
    WindowMode,
)


def test_window_labeling_strategies_read_the_original_window_labels() -> None:
    labels = np.array([0, 1, 0])

    assert WindowLabelingStrategy.OR_POOLING.label(labels)
    assert not WindowLabelingStrategy.START.label(labels)
    assert not WindowLabelingStrategy.END.label(labels)


def test_window_mode_is_derived_from_length_and_stride() -> None:
    assert Windower(length=4, stride=1).mode is WindowMode.OVERLAPPING
    assert Windower(length=4).mode is WindowMode.CONTIGUOUS
    assert Windower(length=4, stride=5).mode is WindowMode.NON_OVERLAPPING


def test_overlapping_windows_are_views_with_point_coverage() -> None:
    series = TimeSeries(np.arange(8), labels=np.array([0, 1, 0, 0, 1, 0, 0, 0]))

    windows = Windower(length=4, stride=2).transform(series)

    assert windows.values.shape == (3, 4, 1)
    assert windows.starts.tolist() == [0, 2, 4]
    assert windows.stops.tolist() == [4, 6, 8]
    assert windows.point_coverage().tolist() == [1, 1, 2, 2, 2, 2, 1, 1]
    assert np.shares_memory(windows.values, series.values)
    assert series.labels is not None
    assert series.labels.tolist() == [
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        False,
    ]


def test_default_stride_produces_non_overlapping_complete_windows() -> None:
    windows = Windower(length=3).transform(TimeSeries(np.arange(8)))

    assert windows.starts.tolist() == [0, 3]
    assert windows.values[:, :, 0].tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert windows.point_coverage().tolist() == [1, 1, 1, 1, 1, 1, 0, 0]


def test_short_series_produces_no_windows() -> None:
    windows = Windower(length=4).transform(TimeSeries(np.arange(3)))

    assert windows.values.shape == (0, 4, 1)
    assert windows.starts.size == 0
    assert windows.point_coverage().tolist() == [0, 0, 0]


def test_edge_padding_preserves_the_original_window_extent() -> None:
    windows = Windower(length=4, tail=TailPolicy.EDGE_PAD).transform(
        TimeSeries(np.arange(6))
    )

    assert windows.starts.tolist() == [0, 4]
    assert windows.valid_lengths is not None
    assert windows.valid_lengths.tolist() == [4, 2]
    assert windows.values[1, :, 0].tolist() == [4.0, 5.0, 5.0, 5.0]
    assert windows.point_coverage().tolist() == [1, 1, 1, 1, 1, 1]


@given(
    n_timesteps=st.integers(min_value=1, max_value=100),
    length=st.integers(min_value=1, max_value=40),
    stride=st.integers(min_value=1, max_value=40),
)
def test_window_geometry_matches_the_complete_window_rule(
    n_timesteps: int, length: int, stride: int
) -> None:
    windows = Windower(length=length, stride=stride).transform(
        TimeSeries(np.arange(n_timesteps))
    )

    expected = 0 if n_timesteps < length else 1 + (n_timesteps - length) // stride
    assert windows.n_windows == expected
    assert np.all(windows.starts + length <= n_timesteps)
    assert np.all(windows.point_coverage() >= 0)
