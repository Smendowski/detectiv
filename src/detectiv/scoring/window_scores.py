from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from detectiv.time_series.windowing import WindowReference


class WindowEvidenceBatch(ABC):
    """Evidence values paired with their original time-series windows."""

    references: tuple[WindowReference, ...]

    def indices_for_series(self, series_id: str) -> list[int]:
        """Return positions of windows belonging to one source series."""
        return [
            index
            for index, reference in enumerate(self.references)
            if reference.series_id == series_id
        ]

    @abstractmethod
    def for_series(self, series_id: str) -> WindowEvidenceBatch:
        """Return evidence restricted to one source series."""
        raise NotImplementedError


@dataclass(frozen=True)
class WindowScoreBatch(WindowEvidenceBatch):
    """One finite scalar evidence value per window."""

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

    def for_series(self, series_id: str) -> WindowScoreBatch:
        """Return scalar window evidence for one source series."""
        indices = self.indices_for_series(series_id)
        return WindowScoreBatch(
            self.values[indices],
            tuple(self.references[index] for index in indices),
        )


@dataclass(frozen=True)
class WindowPointScoreBatch(WindowEvidenceBatch):
    """Finite pointwise evidence values for every valid window position."""

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

    def for_series(self, series_id: str) -> WindowPointScoreBatch:
        """Return pointwise window evidence for one source series."""
        indices = self.indices_for_series(series_id)
        return WindowPointScoreBatch(
            tuple(self.values[index] for index in indices),
            tuple(self.references[index] for index in indices),
        )
