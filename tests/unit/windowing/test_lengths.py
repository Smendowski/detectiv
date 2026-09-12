import numpy as np
import pytest

from detectiv.data import TimeSeries
from detectiv.windowing import ACFWindowLength, FixedWindowLength


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
