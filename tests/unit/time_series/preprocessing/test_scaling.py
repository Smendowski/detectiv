import numpy as np
import pytest

from detectiv.time_series import TimeSeries
from detectiv.time_series.preprocessing import MinMaxScaling


def test_min_max_scaling_fits_training_data_and_does_not_clip_test_values() -> None:
    train = TimeSeries(
        np.array([[0.0, 10.0], [10.0, 30.0]]),
        labels=np.array([0, 0]),
        series_id="series",
    )
    test = TimeSeries(
        np.array([[20.0, 50.0]]),
        labels=np.array([1]),
        series_id="series",
    )

    scaling = MinMaxScaling().fit(train)
    scaled_train = scaling.transform(train)
    scaled_test = scaling.transform(test)

    np.testing.assert_array_equal(
        scaled_train.values,
        np.array([[0.0, 0.0], [1.0, 1.0]]),
    )
    np.testing.assert_array_equal(
        scaled_test.values,
        np.array([[2.0, 2.0]]),
    )
    assert test.labels is not None
    assert scaled_test.labels is not None
    np.testing.assert_array_equal(scaled_test.labels, test.labels)


def test_min_max_scaling_rejects_reordered_named_features() -> None:
    train = TimeSeries(
        np.array([[0.0, 10.0], [10.0, 30.0]]),
        feature_names=("first", "second"),
        series_id="series",
    )
    reordered = TimeSeries(
        np.array([[50.0, 20.0]]),
        feature_names=("second", "first"),
        series_id="series",
    )

    with pytest.raises(ValueError, match="feature names"):
        MinMaxScaling().fit(train).transform(reordered)
