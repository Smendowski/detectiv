from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler

from detectiv.time_series import TimeSeries, TimeSeriesDataset
from detectiv.time_series.preprocessing.base import TimeSeriesPreprocessor


class MinMaxScaling(TimeSeriesPreprocessor):
    """Scale every feature to the range fitted from training data."""

    def __init__(self) -> None:
        """Create an unfitted min-max scaler."""
        self._scaler: SklearnMinMaxScaler | None = None
        self._feature_names: tuple[str, ...] | None = None

    def fit(self, train: TimeSeriesDataset) -> MinMaxScaling:
        """Fit per-feature minima and maxima across all training-series timesteps."""
        values = tuple(train[series_id].values for series_id in train.series_ids)
        n_features = values[0].shape[1]
        if any(series.shape[1] != n_features for series in values):
            raise ValueError("all series must have the same number of features")
        self._scaler = SklearnMinMaxScaler().fit(np.concatenate(values))
        self._feature_names = train[train.series_ids[0]].feature_names
        return self

    def transform(self, dataset: TimeSeriesDataset) -> TimeSeriesDataset:
        """Scale a compatible dataset without clipping out-of-range values.

        Raises:
            RuntimeError: If called before fitting.
            ValueError: If feature names differ from the fitted dataset.
        """
        if self._scaler is None:
            raise RuntimeError("MinMaxScaling must be fitted before transforming data")
        if dataset[dataset.series_ids[0]].feature_names != self._feature_names:
            raise ValueError("dataset feature names do not match fitted data")
        return TimeSeriesDataset(
            dataset.dataset_id,
            {
                series_id: self._transform_series(dataset[series_id])
                for series_id in dataset.series_ids
            },
            metadata=dataset.metadata,
        )

    def _transform_series(self, series: TimeSeries) -> TimeSeries:
        assert self._scaler is not None
        return TimeSeries(
            self._scaler.transform(series.values),
            labels=series.labels,
            feature_names=series.feature_names,
            sampling_rate=series.sampling_rate,
            series_id=series.series_id,
        )
