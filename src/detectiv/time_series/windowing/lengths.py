from abc import ABC, abstractmethod

import numpy as np
from statsmodels.tsa.stattools import acf

from detectiv.time_series import TimeSeries
from detectiv.utils import integer


class WindowLengthStrategy(ABC):
    """Base contract for selecting a window length from a series."""

    @abstractmethod
    def select(self, series: TimeSeries) -> int:
        """Select a positive window length for ``series``."""
        raise NotImplementedError


class FixedWindowLength(WindowLengthStrategy):
    """Always select one validated fixed window length."""

    def __init__(self, length: int) -> None:
        """Create a strategy that always returns ``length``."""
        length = integer(length, "length")
        if length <= 0:
            raise ValueError("length must be positive")
        self.length = length

    def select(self, series: TimeSeries) -> int:
        """Return the configured length without inspecting the series."""
        return self.length


class ACFWindowLength(WindowLengthStrategy):
    """Select a bounded length from the first insignificant autocorrelation lag."""

    def __init__(
        self,
        *,
        feature_index: int = 0,
        min_length: int = 10,
        max_length: int = 100,
        threshold_multiplier: float = 1.96,
    ) -> None:
        """Configure the feature, bounds, and finite threshold multiplier."""
        feature_index = integer(feature_index, "feature_index")
        min_length = integer(min_length, "min_length")
        max_length = integer(max_length, "max_length")
        if feature_index < 0:
            raise ValueError("feature_index must be non-negative")
        if min_length <= 0:
            raise ValueError("min_length must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if max_length < min_length:
            raise ValueError("max_length must not be less than min_length")
        if not np.isfinite(threshold_multiplier) or threshold_multiplier <= 0:
            raise ValueError("threshold_multiplier must be finite and positive")
        self.feature_index = feature_index
        self.min_length = min_length
        self.max_length = max_length
        self.threshold_multiplier = threshold_multiplier

    def select(self, series: TimeSeries) -> int:
        """Select a bounded length from the configured feature's autocorrelation.

        Raises:
            ValueError: If the configured feature does not exist in ``series``.
        """
        if self.feature_index >= series.n_features:
            raise ValueError("feature_index must select a series feature")
        n_timesteps = series.n_timesteps
        max_lag = min(self.max_length, n_timesteps // 4)
        if max_lag < 1:
            return self.min_length

        values = series.values[:, self.feature_index]
        acf_values = acf(values, nlags=max_lag, fft=True)
        threshold = self.threshold_multiplier / np.sqrt(n_timesteps)
        candidates = np.flatnonzero(np.abs(acf_values[1:]) < threshold)
        length = int(candidates[0] + 1) if len(candidates) else max_lag // 2
        return int(np.clip(length, self.min_length, self.max_length))
