from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from detectiv.datasets import TimeSeriesDataset


class ChannelizationContext:
    pass


class TimestampContext(ChannelizationContext):
    pass


class FeatureSelectionContext(ChannelizationContext, ABC):
    @abstractmethod
    def fit_feature_indices(
        self,
        select: Callable[[np.ndarray], tuple[int, ...]],
        train: TimeSeriesDataset,
    ) -> tuple[int, ...] | None:
        raise NotImplementedError

    @abstractmethod
    def feature_indices(
        self,
        select: Callable[[np.ndarray], tuple[int, ...]],
        window: np.ndarray,
        fitted_indices: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        raise NotImplementedError


class WindowContext(FeatureSelectionContext):
    def fit_feature_indices(
        self,
        select: Callable[[np.ndarray], tuple[int, ...]],
        train: TimeSeriesDataset,
    ) -> None:
        return None

    def feature_indices(
        self,
        select: Callable[[np.ndarray], tuple[int, ...]],
        window: np.ndarray,
        fitted_indices: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        return select(window)


class TrainingSetContext(FeatureSelectionContext):
    def fit_feature_indices(
        self,
        select: Callable[[np.ndarray], tuple[int, ...]],
        train: TimeSeriesDataset,
    ) -> tuple[int, ...]:
        return select(self.values(train))

    def feature_indices(
        self,
        select: Callable[[np.ndarray], tuple[int, ...]],
        window: np.ndarray,
        fitted_indices: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        if fitted_indices is None:
            raise RuntimeError("training-set channelization must be fitted")
        return fitted_indices

    def values(self, train: TimeSeriesDataset) -> np.ndarray:
        values = tuple(train[series_id].values for series_id in train.series_ids)
        n_features = values[0].shape[1]
        if any(series.shape[1] != n_features for series in values):
            raise ValueError("training-set context requires a consistent feature count")
        return np.concatenate(values)
