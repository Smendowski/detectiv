from enum import StrEnum
from typing import Self

import numpy as np

from detectiv.time_series import TimeSeriesDataset
from detectiv.ts2i.channelization.base import Channelization


class FeatureSelectionScope(StrEnum):
    WINDOW = "window"
    TRAINING = "training"


class HighestVariabilityFeatureChannelization(Channelization):
    def __init__(
        self,
        n_features: int,
        scope: FeatureSelectionScope | str = FeatureSelectionScope.TRAINING,
    ) -> None:
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        self.n_features = n_features
        self.scope = FeatureSelectionScope(scope)
        self._feature_indices: tuple[int, ...] | None = None

    @property
    def feature_indices(self) -> tuple[int, ...] | None:
        return self._feature_indices

    def fit(self, train: TimeSeriesDataset) -> Self:
        if self.scope is FeatureSelectionScope.TRAINING:
            values = np.concatenate(
                tuple(train[series_id].values for series_id in train.series_ids)
            )
            self._feature_indices = self._select(values)
        return self

    def transform(self, window: np.ndarray) -> tuple[np.ndarray, ...]:
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
