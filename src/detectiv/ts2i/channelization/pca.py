from typing import Self

import numpy as np
from sklearn.decomposition import PCA as SklearnPCA

from detectiv.time_series import TimeSeries
from detectiv.ts2i.channelization.base import Channelization


class PCAChannelization(Channelization):
    """Fit PCA on training observations and emit component time series."""

    def __init__(self, n_components: int) -> None:
        """Configure the positive number of principal components to retain.

        Args:
            n_components: Number of PCA components emitted as planes.

        Raises:
            ValueError: If ``n_components`` is not positive.
        """
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        self.n_components = n_components
        self._model: SklearnPCA | None = None

    def fit(self, train: TimeSeries) -> Self:
        """Fit PCA using the training observations.

        Args:
            train: Training-only source series.

        Returns:
            This fitted channelization.

        Raises:
            ValueError: If there are too few samples or features.
        """
        values = train.values
        if self.n_components > min(values.shape):
            raise ValueError(
                "n_components must not exceed the number of training samples "
                "or features"
            )
        self._model = SklearnPCA(n_components=self.n_components, svd_solver="full")
        self._model.fit(values)
        return self

    def transform(self, window: np.ndarray) -> tuple[np.ndarray, ...]:
        """Project a window and return one plane per fitted component.

        Args:
            window: Time-major feature values to project.

        Returns:
            One component time series per configured component.

        Raises:
            RuntimeError: If called before fitting.
        """
        if self._model is None:
            raise RuntimeError("PCA channelization must be fitted")
        components = self._model.transform(window).astype(np.float32)
        return tuple(components[:, index] for index in range(self.n_components))
