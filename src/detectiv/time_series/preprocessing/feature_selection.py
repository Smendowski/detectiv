from __future__ import annotations

import numpy as np

from detectiv.time_series import TimeSeries
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

    def fit(self, train: TimeSeries) -> ConstantFeatureRemoval:
        """Fit a retained-feature mask from all training-series timesteps.

        Raises:
            ValueError: If every feature is constant within the configured tolerance.
        """
        values = np.asarray(train.values, dtype=np.float64)
        mask = np.ptp(values, axis=0) > self.tolerance
        if not np.any(mask):
            raise ValueError("constant-feature removal would remove every feature")
        self._mask = mask
        self._feature_names = train.feature_names
        return self

    def transform(self, series: TimeSeries) -> TimeSeries:
        """Remove training-constant features from a schema-compatible dataset.

        Raises:
            RuntimeError: If called before fitting.
            ValueError: If feature names differ from the fitted dataset.
        """
        if self._mask is None:
            raise RuntimeError(
                "ConstantFeatureRemoval must be fitted before transforming"
            )
        if series.feature_names != self._feature_names:
            raise ValueError("series feature names do not match fitted data")
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
        return series.with_values(
            series.values[:, self._mask],
            feature_names=feature_names,
        )
