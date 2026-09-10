import numpy as np

from detectiv.data.split import TemporalSplit


class TimeSeries:
    def __init__(
        self,
        values: np.ndarray,
        *,
        labels: np.ndarray | None = None,
        feature_names: tuple[str, ...] | list[str] | None = None,
        sampling_rate: float | None = None,
        series_id: str | None = None,
    ) -> None:
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
        self.values = np.array(values, dtype=dtype, order="C", copy=True)
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

    @property
    def n_timesteps(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])

    @property
    def is_univariate(self) -> bool:
        return self.n_features == 1

    def split(
        self, train_end: int, validation_end: int | None = None
    ) -> TemporalSplit["TimeSeries"]:
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

        return TemporalSplit(
            train=self._segment(0, train_end),
            validation=validation,
            test=self._segment(test_start, self.n_timesteps),
        )

    def _segment(self, start: int, stop: int) -> "TimeSeries":
        labels = None if self.labels is None else self.labels[start:stop]
        return TimeSeries(
            self.values[start:stop],
            labels=labels,
            feature_names=self.feature_names,
            sampling_rate=self.sampling_rate,
            series_id=self.series_id,
        )
