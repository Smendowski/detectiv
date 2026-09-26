from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from detectiv.time_series.split import TemporalSplit
from detectiv.time_series.splitters import TemporalBoundary, TemporalHoldout

if TYPE_CHECKING:
    from detectiv.time_series.preprocessing import TimeSeriesPreprocessor
    from detectiv.time_series.windowing import WindowSpec
    from detectiv.time_series.windowing.split import WindowedTimeSeriesSplit


class TimeSeries:
    """Validated immutable time-series values with optional labels and metadata."""

    def __init__(
        self,
        values: np.ndarray,
        *,
        labels: np.ndarray | None = None,
        feature_names: tuple[str, ...] | list[str] | None = None,
        sampling_rate: float | None = None,
        series_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Create a series from finite real values.

        Args:
            values: One feature vector or a two-dimensional time-by-feature array.
            labels: Optional binary anomaly labels, one per timestep.
            feature_names: Optional unique names for the feature columns.
            sampling_rate: Optional positive sampling rate.
            series_id: Optional identifier retained by derived series.
            metadata: Optional immutable provenance and descriptive metadata.
        """
        values = np.asarray(values)
        if values.ndim == 1:
            values = values[:, np.newaxis]
        if values.ndim != 2:
            raise ValueError(
                "values must have shape (n_timesteps,) or (n_timesteps, n_features)"
            )
        if values.shape[0] == 0:
            raise ValueError("values must contain at least one timestep")
        if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
            values.dtype, np.complexfloating
        ):
            raise TypeError("values must be real-valued numeric data")
        if not np.all(np.isfinite(values)):
            raise ValueError("values must not contain NaN or infinity")

        dtype = np.float32 if values.dtype == np.float32 else np.float64
        with np.errstate(over="ignore"):
            normalized_values = np.array(values, dtype=dtype, order="C", copy=True)
        if not np.all(np.isfinite(normalized_values)):
            raise ValueError("values must remain finite after normalization")
        self.values = normalized_values
        self.values.setflags(write=False)

        self.labels: np.ndarray | None = None
        if labels is not None:
            labels = np.asarray(labels)
            if labels.ndim != 1 or labels.shape[0] != self.n_timesteps:
                raise ValueError("labels must have shape (n_timesteps,)")
            if not np.issubdtype(labels.dtype, np.bool_) and not np.all(
                (labels == 0) | (labels == 1)
            ):
                raise ValueError("labels must contain only binary values")
            self.labels = np.array(labels, dtype=bool, order="C", copy=True)
            self.labels.setflags(write=False)

        self.feature_names: tuple[str, ...] | None = None
        if feature_names is not None:
            names = tuple(feature_names)
            if len(names) != self.n_features or len(set(names)) != len(names):
                raise ValueError("feature_names must be unique and match n_features")
            self.feature_names = names

        if sampling_rate is not None and (
            not np.isfinite(sampling_rate) or sampling_rate <= 0
        ):
            raise ValueError("sampling_rate must be finite and positive")
        self.sampling_rate = sampling_rate
        self.series_id = series_id
        self.metadata = MappingProxyType(dict(metadata or {}))

    @property
    def n_timesteps(self) -> int:
        """Return the number of temporal observations."""
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        """Return the number of feature columns."""
        return int(self.values.shape[1])

    @property
    def is_univariate(self) -> bool:
        """Return whether the series has exactly one feature."""
        return self.n_features == 1

    def with_values(
        self,
        values: np.ndarray,
        *,
        feature_names: tuple[str, ...] | list[str] | None,
    ) -> TimeSeries:
        """Return transformed values while preserving labels and provenance.

        Args:
            values: Replacement time-by-feature values.
            feature_names: Names describing the replacement features, or ``None``
                when they are unnamed.

        Returns:
            A validated derived series.
        """
        normalized_values = np.asarray(values)
        if (
            normalized_values.ndim not in (1, 2)
            or len(normalized_values) != self.n_timesteps
        ):
            raise ValueError("replacement values must preserve the number of timesteps")
        return TimeSeries(
            values,
            labels=self.labels,
            feature_names=feature_names,
            sampling_rate=self.sampling_rate,
            series_id=self.series_id,
            metadata=self.metadata,
        )

    def split(self, rule: TemporalBoundary | TemporalHoldout) -> TimeSeriesSplit:
        """Split the series into chronological train, validation, and test segments.

        Args:
            rule: Concrete temporal boundary or trailing holdout rule.

        Returns:
            Chronological train, optional validation, and test series.

        Raises:
            ValueError: If the resolved boundaries are not strictly internal.
        """
        boundary = rule.boundary() if isinstance(rule, TemporalHoldout) else rule
        train_end = boundary.train_end
        validation_end = boundary.validation_end
        test_start = train_end if validation_end is None else validation_end
        if not 0 < train_end < self.n_timesteps:
            raise ValueError("train_end must lie strictly within the series")
        if (
            validation_end is not None
            and not train_end < validation_end < self.n_timesteps
        ):
            raise ValueError(
                "validation_end must lie strictly between train_end and series end"
            )

        validation = None
        if validation_end is not None:
            validation = self._segment(train_end, validation_end)

        return TimeSeriesSplit(
            train=self._segment(0, train_end),
            validation=validation,
            test=self._segment(test_start, self.n_timesteps),
        )

    def _segment(self, start: int, stop: int) -> TimeSeries:
        labels = None if self.labels is None else self.labels[start:stop]
        return TimeSeries(
            self.values[start:stop],
            labels=labels,
            feature_names=self.feature_names,
            sampling_rate=self.sampling_rate,
            series_id=self.series_id,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class TimeSeriesSplit(TemporalSplit[TimeSeries]):
    """Temporal series partitions ready for optional preprocessing and windowing."""

    preprocessors: tuple[TimeSeriesPreprocessor, ...] = field(
        default=(), repr=False, compare=False
    )

    def preprocess(self, preprocessor: TimeSeriesPreprocessor) -> TimeSeriesSplit:
        """Append train-fitted preprocessing to this split.

        Args:
            preprocessor: Transformer to fit on the training partition later.

        Returns:
            A new immutable split stage carrying all preprocessors in declaration
            order.
        """
        return replace(self, preprocessors=(*self.preprocessors, preprocessor))

    def window(
        self,
        *,
        train: WindowSpec,
        test: WindowSpec,
        validation: WindowSpec | None = None,
    ) -> WindowedTimeSeriesSplit:
        """Configure independent window specifications for each partition.

        Args:
            train: Window specification for the training partition.
            test: Window specification for the test partition.
            validation: Optional specification for the validation partition.

        Returns:
            A windowed stage ready for projection configuration.
        """
        from detectiv.time_series.windowing.split import (
            SplitWindowing,
            WindowedTimeSeriesSplit,
        )

        return WindowedTimeSeriesSplit(
            split=self,
            windowing=SplitWindowing(
                train=train,
                validation=validation,
                test=test,
            ),
        )
