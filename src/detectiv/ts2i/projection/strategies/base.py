from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from detectiv.datasets import TimeSeriesDataset


class ProjectionStrategy(ABC):
    @property
    @abstractmethod
    def n_channels(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def fit(self, train: TimeSeriesDataset) -> Self:
        raise NotImplementedError

    @abstractmethod
    def render(
        self,
        window: np.ndarray,
        size: tuple[int, int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        raise NotImplementedError
