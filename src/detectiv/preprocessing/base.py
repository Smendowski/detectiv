from abc import ABC, abstractmethod
from typing import Self

from detectiv.datasets import TimeSeriesDataset


class TimeSeriesPreprocessor(ABC):
    @abstractmethod
    def fit(self, train: TimeSeriesDataset) -> Self:
        raise NotImplementedError

    @abstractmethod
    def transform(self, dataset: TimeSeriesDataset) -> TimeSeriesDataset:
        raise NotImplementedError
