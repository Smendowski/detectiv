import numpy as np
import pytest

from detectiv.time_series import TimeSeries
from detectiv.time_series.windowing import ACFWindowLength, FixedWindowLength


def test_fixed_window_length_is_independent_of_series_values() -> None:
    strategy = FixedWindowLength(20)

    assert strategy.select(TimeSeries(np.arange(5))) == 20


def test_acf_window_length_uses_the_spiral_bounds() -> None:
    strategy = ACFWindowLength()

    length = strategy.select(TimeSeries(np.random.default_rng(0).normal(size=200)))

    assert 10 <= length <= 100


def test_acf_window_length_selects_an_explicit_feature() -> None:
    strategy = ACFWindowLength(feature_index=1)
    values = np.random.default_rng(1).normal(size=(80, 2))

    length = strategy.select(TimeSeries(values))

    assert 10 <= length <= 100


def test_acf_window_length_rejects_missing_feature() -> None:
    with pytest.raises(ValueError, match="feature_index"):
        ACFWindowLength(feature_index=1).select(TimeSeries(np.arange(80)))


@pytest.mark.parametrize("length", [2.5, True])
def test_fixed_window_length_requires_an_integer(length: object) -> None:
    with pytest.raises(ValueError, match="length must be an integer"):
        FixedWindowLength(length)  # type: ignore[arg-type]


def test_acf_window_length_requires_integral_bounds() -> None:
    with pytest.raises(ValueError, match="max_length must be an integer"):
        ACFWindowLength(max_length=20.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [float("nan"), float("inf")])
def test_acf_window_length_requires_a_finite_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ACFWindowLength(threshold_multiplier=threshold)
