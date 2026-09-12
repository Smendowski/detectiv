import numpy as np

from detectiv.data import TimeSeries
from detectiv.datasets import TimeSeriesDataset
from detectiv.preprocessing.base import TimeSeriesPreprocessor


class ConstantFeatureRemoval(TimeSeriesPreprocessor):
    def __init__(self, *, tolerance: float = 0.0) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self.tolerance = tolerance
        self._mask: np.ndarray | None = None

    def fit(self, train: TimeSeriesDataset) -> "ConstantFeatureRemoval":
        values = tuple(train[series_id].values for series_id in train.series_ids)
        n_features = values[0].shape[1]
        if any(series.shape[1] != n_features for series in values):
            raise ValueError("all series must have the same number of features")
        self._mask = np.ptp(np.concatenate(values), axis=0) > self.tolerance
        if not np.any(self._mask):
            raise ValueError("constant-feature removal would remove every feature")
        return self

    def transform(self, dataset: TimeSeriesDataset) -> TimeSeriesDataset:
        if self._mask is None:
            raise RuntimeError(
                "ConstantFeatureRemoval must be fitted before transforming"
            )
        return TimeSeriesDataset(
            dataset.dataset_id,
            {
                series_id: self._transform_series(dataset[series_id])
                for series_id in dataset.series_ids
            },
            metadata=dataset.metadata,
        )

    def _transform_series(self, series: TimeSeries) -> TimeSeries:
        if self._mask is None:
            raise RuntimeError(
                "ConstantFeatureRemoval must be fitted before transforming"
            )
        if series.n_features != len(self._mask):
            raise ValueError("series feature count does not match fitted data")
        feature_names = None
        if series.feature_names is not None:
            feature_names = tuple(
                name
                for name, keep in zip(series.feature_names, self._mask, strict=True)
                if keep
            )
        return TimeSeries(
            series.values[:, self._mask],
            labels=series.labels,
            feature_names=feature_names,
            sampling_rate=series.sampling_rate,
            series_id=series.series_id,
        )
