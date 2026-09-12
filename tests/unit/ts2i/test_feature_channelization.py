import numpy as np

from detectiv.ts2i.channelization import FeatureChannelization
from detectiv.ts2i.channelization.context import WindowContext


def test_feature_channelization_returns_selected_feature_series() -> None:
    channels = FeatureChannelization((2, 0), WindowContext()).transform(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )

    np.testing.assert_array_equal(channels[0], [3.0, 6.0])
    np.testing.assert_array_equal(channels[1], [1.0, 4.0])
