from typing import Self

import numpy as np

from detectiv.datasets import TimeSeriesDataset
from detectiv.ts2i.channelization.base import Channelization
from detectiv.ts2i.channelization.context import FeatureSelectionContext


class HighestVariabilityFeaturesChannelization(Channelization):
    def __init__(self, n_features: int, context: FeatureSelectionContext) -> None:
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        super().__init__(context)
        self._selection_context = context
        self.n_features = n_features
        self._feature_indices: tuple[int, ...] | None = None

    @property
    def feature_indices(self) -> tuple[int, ...] | None:
        return self._feature_indices

    def fit(self, train: TimeSeriesDataset) -> Self:
        self._feature_indices = self._selection_context.fit_feature_indices(
            self._select,
            train,
        )
        return self

    def transform(self, window: np.ndarray) -> tuple[np.ndarray, ...]:
        if window.ndim != 2:
            raise ValueError("feature selection requires a two-dimensional window")
        indices = self._selection_context.feature_indices(
            self._select,
            window,
            self._feature_indices,
        )
        return tuple(window[:, index] for index in indices)

    def _select(self, values: np.ndarray) -> tuple[int, ...]:
        if values.ndim != 2 or self.n_features > values.shape[1]:
            raise ValueError("n_features must select available input features")
        variability = np.var(values, axis=0)
        indices = np.arange(values.shape[1])
        order = np.lexsort((indices, -variability))
        return tuple(int(index) for index in order[: self.n_features])
