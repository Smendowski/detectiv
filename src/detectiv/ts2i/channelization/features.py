from collections.abc import Sequence

import numpy as np

from detectiv.ts2i.channelization.base import Channelization


class IndexedFeatureChannelization(Channelization):
    """Extract selected input features as one univariate plane each."""

    def __init__(self, feature_indices: Sequence[int]) -> None:
        """Configure unique zero-based feature indices to extract.

        Args:
            feature_indices: Non-empty unique indices into window features.

        Raises:
            ValueError: If indices are empty, negative, or duplicated.
        """
        indices = tuple(feature_indices)
        if not indices or any(index < 0 for index in indices):
            raise ValueError("feature_indices must contain non-negative indices")
        if len(set(indices)) != len(indices):
            raise ValueError("feature_indices must be unique")
        self.feature_indices = indices

    def transform(self, window: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return one time-value plane for every configured feature.

        Args:
            window: Two-dimensional time-major feature values.

        Returns:
            Selected features in configured order.

        Raises:
            ValueError: If the window is not two-dimensional or lacks a selected
                feature.
        """
        if window.ndim != 2:
            raise ValueError("feature channelization requires a two-dimensional window")
        if max(self.feature_indices) >= window.shape[1]:
            raise ValueError("feature_indices must select window features")
        return tuple(window[:, index] for index in self.feature_indices)
