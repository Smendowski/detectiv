from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from detectiv.data import WindowReference


class WindowEvidenceBatch(ABC):
    references: tuple[WindowReference, ...]

    def indices_for_series(self, series_id: str) -> list[int]:
        return [
            index
            for index, reference in enumerate(self.references)
            if reference.series_id == series_id
        ]

    @abstractmethod
    def for_series(self, series_id: str) -> "WindowEvidenceBatch":
        raise NotImplementedError


@dataclass(frozen=True)
class WindowScoreBatch(WindowEvidenceBatch):
    values: np.ndarray
    references: tuple[WindowReference, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("window scores must be one-dimensional")
        if len(values) != len(self.references):
            raise ValueError("window scores and references must have the same length")
        if not np.isfinite(values).all():
            raise ValueError("window scores must be finite")
        values = np.array(values, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    def for_series(self, series_id: str) -> "WindowScoreBatch":
        indices = self.indices_for_series(series_id)
        return WindowScoreBatch(
            self.values[indices],
            tuple(self.references[index] for index in indices),
        )


@dataclass(frozen=True)
class WindowPointScoreBatch(WindowEvidenceBatch):
    values: tuple[np.ndarray, ...]
    references: tuple[WindowReference, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.references):
            raise ValueError("window scores and references must have the same length")
        point_scores: list[np.ndarray] = []
        for values, reference in zip(self.values, self.references, strict=True):
            values = np.asarray(values, dtype=np.float64)
            if values.ndim != 1 or len(values) != reference.valid_length:
                raise ValueError("point scores must match each window's valid length")
            if not np.isfinite(values).all():
                raise ValueError("point scores must be finite")
            values = np.array(values, copy=True)
            values.setflags(write=False)
            point_scores.append(values)
        object.__setattr__(self, "values", tuple(point_scores))

    def for_series(self, series_id: str) -> "WindowPointScoreBatch":
        indices = self.indices_for_series(series_id)
        return WindowPointScoreBatch(
            tuple(self.values[index] for index in indices),
            tuple(self.references[index] for index in indices),
        )


@dataclass(frozen=True)
class WindowSaliencyBatch(WindowEvidenceBatch):
    values: np.ndarray
    saliencies: tuple[np.ndarray, ...]
    references: tuple[WindowReference, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("window scores must be one-dimensional")
        if len(values) != len(self.references) or len(values) != len(self.saliencies):
            raise ValueError(
                "scores, saliencies, and references must have the same length"
            )
        if not np.isfinite(values).all():
            raise ValueError("window scores must be finite")

        saliencies: list[np.ndarray] = []
        for saliency, reference in zip(self.saliencies, self.references, strict=True):
            saliency = np.asarray(saliency, dtype=np.float64)
            if saliency.ndim != 1 or len(saliency) != reference.valid_length:
                raise ValueError("saliencies must match each window's valid length")
            if not np.isfinite(saliency).all() or (saliency < 0).any():
                raise ValueError("saliencies must be finite and non-negative")
            if saliency.mean() == 0:
                raise ValueError("each window saliency must contain a positive value")
            saliency = np.array(saliency, copy=True)
            saliency.setflags(write=False)
            saliencies.append(saliency)

        values = np.array(values, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "saliencies", tuple(saliencies))

    def for_series(self, series_id: str) -> "WindowSaliencyBatch":
        indices = self.indices_for_series(series_id)
        return WindowSaliencyBatch(
            self.values[indices],
            tuple(self.saliencies[index] for index in indices),
            tuple(self.references[index] for index in indices),
        )
