from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from sklearn.decomposition import PCA as SklearnPCA

from detectiv.datasets import TimeSeriesDataset


class Channelization(ABC):
    def fit(self, train: TimeSeriesDataset) -> Self:
        return self

    @abstractmethod
    def transform(self, window: np.ndarray) -> tuple[np.ndarray, ...]:
        raise NotImplementedError


class Identity(Channelization):
    def transform(self, window: np.ndarray) -> tuple[np.ndarray, ...]:
        return (window,)


class MSM(Channelization):
    def transform(self, window: np.ndarray) -> tuple[np.ndarray, ...]:
        return (
            window.mean(axis=1),
            window.std(axis=1),
            window.max(axis=1),
        )


class PCA(Channelization):
    def __init__(self, n_components: int) -> None:
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        self.n_components = n_components
        self._model: SklearnPCA | None = None

    def fit(self, train: TimeSeriesDataset) -> "PCA":
        values = tuple(train[series_id].values for series_id in train.series_ids)
        n_features = values[0].shape[1]
        if self.n_components > n_features:
            raise ValueError("n_components must not exceed the number of features")
        if any(series.shape[1] != n_features for series in values):
            raise ValueError("PCA requires the same number of features in every series")
        self._model = SklearnPCA(n_components=self.n_components, svd_solver="full")
        self._model.fit(np.concatenate(values))
        return self

    def transform(self, window: np.ndarray) -> tuple[np.ndarray, ...]:
        if self._model is None:
            raise RuntimeError("PCA must be fitted before transforming windows")
        components = self._model.transform(window).astype(np.float32)
        return tuple(components[:, index] for index in range(self.n_components))
