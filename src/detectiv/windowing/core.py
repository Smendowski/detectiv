from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import numpy as np

from detectiv.data import TimeSeries


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
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.stride is not None and self.stride <= 0:
            raise ValueError("stride must be positive")

    @property
    def resolved_stride(self) -> int:
        return self.length if self.stride is None else self.stride

    def windower(self) -> "Windower":
        return Windower(self.length, self.stride, self.tail)


@dataclass
class WindowBatch:
    values: np.ndarray
    starts: np.ndarray
    series_length: int
    valid_lengths: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.valid_lengths is None:
            self.valid_lengths = np.full(
                self.values.shape[0], self.values.shape[1], dtype=np.intp
            )

    @property
    def n_windows(self) -> int:
        return int(self.values.shape[0])

    @property
    def length(self) -> int:
        return int(self.values.shape[1])

    @property
    def stops(self) -> np.ndarray:
        assert self.valid_lengths is not None
        return cast(np.ndarray, self.starts + self.valid_lengths)

    def point_coverage(self) -> np.ndarray:
        changes = np.zeros(self.series_length + 1, dtype=np.intp)
        np.add.at(changes, self.starts, 1)
        np.add.at(changes, self.stops, -1)
        return np.cumsum(changes[:-1])


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

        tail_start = n_complete * self.stride
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
