from dataclasses import dataclass
from enum import StrEnum
from operator import index
from typing import SupportsIndex, cast

import numpy as np
from numpy.typing import NDArray

from detectiv.time_series.series import TimeSeries


class WindowMode(StrEnum):
    OVERLAPPING = "overlapping"
    CONTIGUOUS = "contiguous"
    NON_OVERLAPPING = "non_overlapping"


class TailPolicy(StrEnum):
    DROP = "drop"
    ZERO_PAD = "zero_pad"
    EDGE_PAD = "edge_pad"


class WindowLabelingStrategy(StrEnum):
    OR_POOLING = "or_pooling"
    START = "start"
    END = "end"

    def label(self, labels: np.ndarray) -> bool:
        if labels.ndim != 1 or not len(labels):
            raise ValueError("labels must be a non-empty one-dimensional array")
        if self is WindowLabelingStrategy.OR_POOLING:
            return bool(labels.any())
        if self is WindowLabelingStrategy.START:
            return bool(labels[0])
        return bool(labels[-1])


@dataclass(frozen=True)
class WindowSpec:
    length: int
    stride: int | None = None
    tail: TailPolicy = TailPolicy.DROP
    labeling_strategy: WindowLabelingStrategy = WindowLabelingStrategy.OR_POOLING

    def __post_init__(self) -> None:
        length = _positive_index(self.length, "length")
        stride = None if self.stride is None else _positive_index(self.stride, "stride")
        if length <= 0:
            raise ValueError("length must be positive")
        if stride is not None and stride <= 0:
            raise ValueError("stride must be positive")
        try:
            tail = TailPolicy(self.tail)
        except ValueError as error:
            raise ValueError(f"unsupported tail policy: {self.tail!r}") from error
        try:
            labeling_strategy = WindowLabelingStrategy(self.labeling_strategy)
        except ValueError as error:
            raise ValueError(
                f"unsupported window labeling strategy: {self.labeling_strategy!r}"
            ) from error
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "stride", stride)
        object.__setattr__(self, "tail", tail)
        object.__setattr__(self, "labeling_strategy", labeling_strategy)

    @property
    def resolved_stride(self) -> int:
        return self.length if self.stride is None else self.stride

    def windower(self) -> "Windower":
        return Windower(self.length, self.stride, self.tail)


def _positive_index(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


@dataclass(frozen=True)
class WindowBatch:
    values: np.ndarray
    starts: np.ndarray
    series_length: int
    valid_lengths: np.ndarray | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.flags.writeable:
            values = np.array(values, copy=True)
        if values.ndim != 3 or values.shape[1] <= 0 or values.shape[2] <= 0:
            raise ValueError(
                "window values must have shape (n_windows, length, n_features)"
            )

        series_length = _index_value(self.series_length, "series_length")
        if series_length <= 0:
            raise ValueError("series_length must be positive")

        starts = _index_array(self.starts, "starts", len(values))
        valid_lengths = (
            np.full(len(values), values.shape[1], dtype=np.intp)
            if self.valid_lengths is None
            else _index_array(self.valid_lengths, "valid_lengths", len(values))
        )

        if (starts < 0).any() or (valid_lengths <= 0).any():
            raise ValueError(
                "window starts must be non-negative and valid lengths positive"
            )
        if (valid_lengths > values.shape[1]).any():
            raise ValueError("window valid lengths must not exceed the window length")
        if (starts + valid_lengths > series_length).any():
            raise ValueError("windows must lie within the original series")

        values.setflags(write=False)
        starts.setflags(write=False)
        valid_lengths.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "series_length", series_length)
        object.__setattr__(self, "valid_lengths", valid_lengths)

    @property
    def n_windows(self) -> int:
        return int(self.values.shape[0])

    @property
    def length(self) -> int:
        return int(self.values.shape[1])

    @property
    def stops(self) -> NDArray[np.intp]:
        assert self.valid_lengths is not None
        stops = self.starts + self.valid_lengths
        stops.setflags(write=False)
        return cast(NDArray[np.intp], stops)

    def point_coverage(self) -> NDArray[np.intp]:
        changes = np.zeros(self.series_length + 1, dtype=np.intp)
        np.add.at(changes, self.starts, 1)
        np.add.at(changes, self.stops, -1)
        return cast(NDArray[np.intp], np.cumsum(changes[:-1]))


def _index_value(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        return index(cast(SupportsIndex, value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _index_array(values: object, name: str, length: int) -> NDArray[np.intp]:
    array = np.asarray(values)
    if (
        array.ndim != 1
        or len(array) != length
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise ValueError(f"{name} must be a one-dimensional integer array per window")
    return cast(NDArray[np.intp], np.array(array, dtype=np.intp, copy=True))


class Windower:
    def __init__(
        self,
        length: int,
        stride: int | None = None,
        tail: TailPolicy = TailPolicy.DROP,
    ) -> None:
        self.spec = WindowSpec(length, stride, tail)

    @property
    def length(self) -> int:
        return self.spec.length

    @property
    def stride(self) -> int:
        return self.spec.resolved_stride

    @property
    def mode(self) -> WindowMode:
        if self.stride < self.length:
            return WindowMode.OVERLAPPING
        if self.stride == self.length:
            return WindowMode.CONTIGUOUS
        return WindowMode.NON_OVERLAPPING

    def transform(self, series: TimeSeries) -> WindowBatch:
        n_complete = 0
        if series.n_timesteps >= self.length:
            n_complete = 1 + (series.n_timesteps - self.length) // self.stride

        if n_complete:
            values = np.lib.stride_tricks.sliding_window_view(
                series.values, self.length, axis=0
            )
            values = np.moveaxis(values[:: self.stride], -1, 1)
            starts = np.arange(n_complete, dtype=np.intp) * self.stride
        else:
            values = np.empty(
                (0, self.length, series.n_features), dtype=series.values.dtype
            )
            starts = np.empty(0, dtype=np.intp)

        tail_start = 0 if not n_complete else int(starts[-1]) + self.length
        if self.spec.tail is TailPolicy.DROP or tail_start >= series.n_timesteps:
            return WindowBatch(values, starts, series.n_timesteps)

        valid_length = series.n_timesteps - tail_start
        tail = np.zeros((1, self.length, series.n_features), dtype=series.values.dtype)
        tail[0, :valid_length] = series.values[tail_start:]
        if self.spec.tail is TailPolicy.EDGE_PAD:
            tail[0, valid_length:] = tail[0, valid_length - 1]

        return WindowBatch(
            values=np.concatenate((values, tail)),
            starts=np.append(starts, tail_start),
            series_length=series.n_timesteps,
            valid_lengths=np.append(
                np.full(n_complete, self.length, dtype=np.intp), valid_length
            ),
        )
