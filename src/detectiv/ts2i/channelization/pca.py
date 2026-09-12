from typing import Self

import numpy as np
from sklearn.decomposition import PCA as SklearnPCA

from detectiv.datasets import TimeSeriesDataset
from detectiv.ts2i.channelization.base import Channelization
from detectiv.ts2i.channelization.context import TrainingSetContext


class PCAChannelization(Channelization):
    def __init__(self, n_components: int, context: TrainingSetContext) -> None:
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        super().__init__(context)
        self._training_context = context
        self.n_components = n_components
        self._model: SklearnPCA | None = None

    def fit(self, train: TimeSeriesDataset) -> Self:
        values = self._training_context.values(train)
        if self.n_components > values.shape[1]:
            raise ValueError("n_components must not exceed the number of features")
        self._model = SklearnPCA(n_components=self.n_components, svd_solver="full")
        self._model.fit(values)
        return self

    def transform(self, window: np.ndarray) -> tuple[np.ndarray, ...]:
        if self._model is None:
            raise RuntimeError("PCA channelization must be fitted")
        components = self._model.transform(window).astype(np.float32)
        return tuple(components[:, index] for index in range(self.n_components))
