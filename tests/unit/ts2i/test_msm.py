import numpy as np

from detectiv.ts2i.channelization import MSMChannelization
from detectiv.ts2i.channelization.context import TimestampContext


def test_msm_uses_only_each_timestamp_feature_vector() -> None:
    msm = MSMChannelization(TimestampContext())

    mean, std, maximum = msm.transform(np.array([[2.0, 2.0]]))

    np.testing.assert_array_equal(mean, [2.0])
    np.testing.assert_array_equal(std, [0.0])
    np.testing.assert_array_equal(maximum, [2.0])
