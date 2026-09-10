import numpy as np

from detectiv.data import TimeSeries
from detectiv.datasets import TimeSeriesDataset
from detectiv.preprocessing import MinMaxScaling


def test_min_max_scaling_fits_training_data_and_does_not_clip_test_values() -> None:
    train = TimeSeriesDataset(
        "train",
        {
            "series": TimeSeries(
                np.array([[0.0, 10.0], [10.0, 30.0]]),
                labels=np.array([0, 0]),
                series_id="series",
            )
        },
    )
    test = TimeSeriesDataset(
        "test",
        {
            "series": TimeSeries(
                np.array([[20.0, 50.0]]),
                labels=np.array([1]),
                series_id="series",
            )
        },
    )

    scaling = MinMaxScaling().fit(train)
    scaled_train = scaling.transform(train)
    scaled_test = scaling.transform(test)

    np.testing.assert_array_equal(
        scaled_train["series"].values,
        np.array([[0.0, 0.0], [1.0, 1.0]]),
    )
    np.testing.assert_array_equal(
        scaled_test["series"].values,
        np.array([[2.0, 2.0]]),
    )
    assert test["series"].labels is not None
    assert scaled_test["series"].labels is not None
    np.testing.assert_array_equal(scaled_test["series"].labels, test["series"].labels)
