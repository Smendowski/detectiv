from __future__ import annotations

from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler

from detectiv.time_series import TimeSeries
from detectiv.time_series.preprocessing.base import TimeSeriesPreprocessor


class MinMaxScaling(TimeSeriesPreprocessor):
    """Scale every feature to the range fitted from training data."""

    def __init__(self) -> None:
        """Create an unfitted min-max scaler."""
        self._scaler: SklearnMinMaxScaler | None = None
        self._feature_names: tuple[str, ...] | None = None

    def fit(self, train: TimeSeries) -> MinMaxScaling:
        """Fit per-feature minima and maxima on the training observations."""
        self._scaler = SklearnMinMaxScaler().fit(train.values)
        self._feature_names = train.feature_names
        return self

    def transform(self, series: TimeSeries) -> TimeSeries:
        """Scale a compatible dataset without clipping out-of-range values.

        Raises:
            RuntimeError: If called before fitting.
            ValueError: If feature names differ from the fitted dataset.
        """
        if self._scaler is None:
            raise RuntimeError("MinMaxScaling must be fitted before transforming data")
        if series.feature_names != self._feature_names:
            raise ValueError("series feature names do not match fitted data")
        return series.with_values(
            self._scaler.transform(series.values),
            feature_names=series.feature_names,
        )
