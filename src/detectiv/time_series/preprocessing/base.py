from abc import ABC, abstractmethod
from typing import Self

from detectiv.time_series import TimeSeries


class TimeSeriesPreprocessor(ABC):
    """Base contract for preprocessors fitted on a training series."""

    @abstractmethod
    def fit(self, train: TimeSeries) -> Self:
        """Fit the preprocessor using only training data."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, series: TimeSeries) -> TimeSeries:
        """Transform a series using previously fitted state."""
        raise NotImplementedError
