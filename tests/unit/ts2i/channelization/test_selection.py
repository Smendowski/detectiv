import numpy as np
import pytest

from detectiv.time_series import TimeSeries, TimeSeriesDataset
from detectiv.ts2i.channelization import (
    FeatureSelectionScope,
    HighestVariabilityFeatureChannelization,
)


def test_selects_most_variable_training_features_in_stable_order() -> None:
    train = TimeSeriesDataset(
        "train",
        {
            "series": TimeSeries(
                np.array([[0.0, 4.0, 1.0], [1.0, 0.0, 1.0], [2.0, 8.0, 1.0]]),
                series_id="series",
            )
        },
    )
    channelization = HighestVariabilityFeatureChannelization(
        2,
        FeatureSelectionScope.TRAINING,
    ).fit(train)

    channels = channelization.transform(
        np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    )

    assert channelization.feature_indices == (1, 0)
    np.testing.assert_array_equal(channels[0], [20.0, 50.0])
    np.testing.assert_array_equal(channels[1], [10.0, 40.0])


def test_training_set_context_requires_fit_before_selecting_features() -> None:
    with pytest.raises(RuntimeError, match="must be fitted"):
        HighestVariabilityFeatureChannelization(
            1,
            FeatureSelectionScope.TRAINING,
        ).transform(np.ones((2, 2)))


def test_training_selection_rejects_windows_with_too_few_features() -> None:
    channelization = HighestVariabilityFeatureChannelization(
        2, FeatureSelectionScope.TRAINING
    ).fit(
        TimeSeriesDataset(
            "train",
            {"series": TimeSeries(np.ones((2, 2)), series_id="series")},
        )
    )

    with pytest.raises(ValueError, match="available input features"):
        channelization.transform(np.ones((2, 1)))


def test_window_context_selects_features_from_each_window() -> None:
    channelization = HighestVariabilityFeatureChannelization(
        1,
        FeatureSelectionScope.WINDOW,
    )

    channels = channelization.transform(np.array([[0.0, 3.0], [1.0, 0.0]]))

    np.testing.assert_array_equal(channels[0], [3.0, 0.0])
