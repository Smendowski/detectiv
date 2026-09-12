from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from detectiv.datasets import TimeSeriesDataset
from detectiv.ts2i.channelization.context import ChannelizationContext


class Channelization(ABC):
    def __init__(self, context: ChannelizationContext) -> None:
        self.context = context

    def fit(self, train: TimeSeriesDataset) -> Self:
        return self

    @abstractmethod
    def transform(self, window: np.ndarray) -> tuple[np.ndarray, ...]:
        raise NotImplementedError


class IdentityChannelization(Channelization):
    def transform(self, window: np.ndarray) -> tuple[np.ndarray, ...]:
        return (window,)
