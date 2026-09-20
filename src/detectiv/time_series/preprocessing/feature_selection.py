from __future__ import annotations

import numpy as np

from detectiv.time_series import TimeSeries, TimeSeriesDataset
from detectiv.time_series.preprocessing.base import TimeSeriesPreprocessor


class ConstantFeatureRemoval(TimeSeriesPreprocessor):
    """Remove features whose pooled training range is within a configured tolerance."""

    def __init__(self, *, tolerance: float = 0.0) -> None:
        """Create an unfitted selector with a finite, non-negative range tolerance."""
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and non-negative")
        self.tolerance = tolerance
        self._mask: np.ndarray | None = None
        self._feature_names: tuple[str, ...] | None = None

    def fit(self, train: TimeSeriesDataset) -> ConstantFeatureRemoval:
        """Fit a retained-feature mask from all training-series timesteps.

        Raises:
            ValueError: If every feature is constant within the configured tolerance.
        """
        values = tuple(train[series_id].values for series_id in train.series_ids)
        n_features = values[0].shape[1]
        if any(series.shape[1] != n_features for series in values):
            raise ValueError("all series must have the same number of features")
        mask = np.ptp(np.concatenate(values), axis=0) > self.tolerance
        if not np.any(mask):
            raise ValueError("constant-feature removal would remove every feature")
        self._mask = mask
        self._feature_names = train[train.series_ids[0]].feature_names
        return self

    def transform(self, dataset: TimeSeriesDataset) -> TimeSeriesDataset:
        """Remove training-constant features from a schema-compatible dataset.

        Raises:
            RuntimeError: If called before fitting.
            ValueError: If feature names differ from the fitted dataset.
        """
        if self._mask is None:
            raise RuntimeError(
                "ConstantFeatureRemoval must be fitted before transforming"
            )
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
        assert self._mask is not None
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
