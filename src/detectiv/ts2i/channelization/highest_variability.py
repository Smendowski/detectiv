from enum import StrEnum
from typing import Self

import numpy as np

from detectiv.time_series import TimeSeriesDataset
from detectiv.ts2i.channelization.base import Channelization


class FeatureSelectionScope(StrEnum):
    """Data used to rank features by variance."""

    WINDOW = "window"
    TRAINING = "training"


class HighestVariabilityFeatureChannelization(Channelization):
    """Select the most variable input features as univariate planes."""

    def __init__(
        self,
        n_features: int,
        scope: FeatureSelectionScope | str = FeatureSelectionScope.TRAINING,
    ) -> None:
        """Configure the number of features and ranking scope.

        Args:
            n_features: Positive number of features selected per window.
            scope: Rank features across training data or independently per window.

        Raises:
            ValueError: If ``n_features`` is not positive.
        """
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        self.n_features = n_features
        self.scope = FeatureSelectionScope(scope)
        self._feature_indices: tuple[int, ...] | None = None

    @property
    def feature_indices(self) -> tuple[int, ...] | None:
        """Return fitted training feature indices, when available.

        Returns:
            Selected indices for training scope, otherwise ``None``.
        """
        return self._feature_indices

    def fit(self, train: TimeSeriesDataset) -> Self:
        """Rank features across training data when using training scope.

        Args:
            train: Training-only source series.

        Returns:
            This channelization, potentially with selected feature indices.
        """
        if self.scope is FeatureSelectionScope.TRAINING:
            values = np.concatenate(
                tuple(train[series_id].values for series_id in train.series_ids)
            )
            self._feature_indices = self._select(values)
        return self

    def transform(self, window: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return the selected feature values as separate planes.

        Args:
            window: Two-dimensional time-major feature values.

        Returns:
            Selected features in descending variability order.

        Raises:
            RuntimeError: If training-scope selection has not been fitted.
            ValueError: If the window cannot provide the requested features.
        """
        if window.ndim != 2:
            raise ValueError("feature selection requires a two-dimensional window")
        if self.scope is FeatureSelectionScope.WINDOW:
            indices = self._select(window)
        else:
            if self._feature_indices is None:
                raise RuntimeError("training feature selection must be fitted")
            indices = self._feature_indices
        if any(index >= window.shape[1] for index in indices):
            raise ValueError("n_features must select available input features")
        return tuple(window[:, index] for index in indices)

    def _select(self, values: np.ndarray) -> tuple[int, ...]:
        if values.ndim != 2 or self.n_features > values.shape[1]:
            raise ValueError("n_features must select available input features")
        variability = np.var(values, axis=0)
        indices = np.arange(values.shape[1])
        order = np.lexsort((indices, -variability))
        return tuple(int(index) for index in order[: self.n_features])
