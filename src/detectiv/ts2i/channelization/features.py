from collections.abc import Sequence

import numpy as np

from detectiv.ts2i.channelization.base import Channelization
from detectiv.ts2i.channelization.context import WindowContext


class FeatureChannelization(Channelization):
    def __init__(
        self,
        feature_indices: Sequence[int],
        context: WindowContext,
    ) -> None:
        indices = tuple(feature_indices)
        if not indices or any(index < 0 for index in indices):
            raise ValueError("feature_indices must contain non-negative indices")
        if len(set(indices)) != len(indices):
            raise ValueError("feature_indices must be unique")
        super().__init__(context)
        self.feature_indices = indices

    def transform(self, window: np.ndarray) -> tuple[np.ndarray, ...]:
        if window.ndim != 2:
            raise ValueError("feature channelization requires a two-dimensional window")
        if max(self.feature_indices) >= window.shape[1]:
            raise ValueError("feature_indices must select window features")
        return tuple(window[:, index] for index in self.feature_indices)
