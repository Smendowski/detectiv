import numpy as np

from detectiv.data import TimeSeries
from detectiv.datasets import TimeSeriesDataset
from detectiv.ts2i.channelization import MSMChannelization, PCAChannelization
from detectiv.ts2i.channelization.context import TimestampContext, TrainingSetContext


def test_msm_creates_mean_standard_deviation_and_maximum_channels() -> None:
    mapped = MSMChannelization(TimestampContext()).transform(
        np.array([[1.0, 3.0], [2.0, 6.0]])
    )

    np.testing.assert_array_equal(mapped[0], [2.0, 4.0])
    np.testing.assert_array_equal(mapped[1], [1.0, 2.0])
    np.testing.assert_array_equal(mapped[2], [3.0, 6.0])


def test_pca_fits_on_training_series_and_exposes_each_component() -> None:
    train = TimeSeriesDataset(
        "train",
        {"series": TimeSeries(np.array([[0.0, 0.0], [1.0, 1.0]]), series_id="series")},
    )
    pca = PCAChannelization(2, TrainingSetContext()).fit(train)

    components = pca.transform(np.array([[2.0, 2.0], [3.0, 3.0]]))

    assert len(components) == 2
    assert all(component.shape == (2,) for component in components)
