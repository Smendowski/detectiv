from abc import ABC, abstractmethod
from typing import Self

from detectiv.time_series.dataset import TimeSeriesDataset


class TimeSeriesPreprocessor(ABC):
    """Base contract for preprocessors fitted on training datasets."""

    @abstractmethod
    def fit(self, train: TimeSeriesDataset) -> Self:
        """Fit the preprocessor using only training data."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, dataset: TimeSeriesDataset) -> TimeSeriesDataset:
        """Transform a dataset using previously fitted state."""
        raise NotImplementedError
