from abc import ABC, abstractmethod

import numpy as np
from statsmodels.tsa.stattools import acf

from detectiv.data import TimeSeries


class WindowLengthStrategy(ABC):
    @abstractmethod
    def select(self, series: TimeSeries) -> int:
        raise NotImplementedError


class FixedWindowLength(WindowLengthStrategy):
    def __init__(self, length: int) -> None:
        if length <= 0:
            raise ValueError("length must be positive")
        self.length = length

    def select(self, series: TimeSeries) -> int:
        return self.length


class ACFWindowLength(WindowLengthStrategy):
    def __init__(
        self,
        *,
        feature_index: int = 0,
        min_length: int = 10,
        max_length: int = 100,
        threshold_multiplier: float = 1.96,
    ) -> None:
        if feature_index < 0 or min_length <= 0 or max_length < min_length:
            raise ValueError("feature index and window-length bounds are invalid")
        if threshold_multiplier <= 0:
            raise ValueError("threshold_multiplier must be positive")
        self.feature_index = feature_index
        self.min_length = min_length
        self.max_length = max_length
        self.threshold_multiplier = threshold_multiplier

    def select(self, series: TimeSeries) -> int:
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
