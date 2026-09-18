from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from detectiv.time_series import TimeSeriesDataset


class Channelization(ABC):
    def fit(self, train: TimeSeriesDataset) -> Self:
        return self

    @abstractmethod
    def transform(self, window: np.ndarray) -> tuple[np.ndarray, ...]:
        raise NotImplementedError


class IdentityChannelization(Channelization):
    def transform(self, window: np.ndarray) -> tuple[np.ndarray, ...]:
        return (window,)
