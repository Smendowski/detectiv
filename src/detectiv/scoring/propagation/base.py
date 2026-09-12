from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import StrEnum

import numpy as np

from detectiv.data import WindowReference
from detectiv.scoring.window_scores import (
    WindowEvidenceBatch,
    WindowPointScoreBatch,
    WindowScoreBatch,
)


class UncoveredPolicy(StrEnum):
    ERROR = "error"
    EDGE_PAD = "edge_pad"


class WindowToPointScorePropagationStrategy(ABC):
    def __init__(self, uncovered: UncoveredPolicy = UncoveredPolicy.ERROR) -> None:
        self.uncovered = uncovered

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def transform(
        self,
        scores: WindowEvidenceBatch,
        series_length: int,
    ) -> np.ndarray:
        if series_length <= 0:
            raise ValueError("series_length must be positive")
        if not scores.references:
            raise ValueError("at least one window score is required")
        point_scores, coverage = self._aggregate(scores, series_length)
        return self._resolve_uncovered(point_scores, coverage)

    @abstractmethod
    def _aggregate(
        self,
        scores: WindowEvidenceBatch,
        series_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def contributions(
        scores: WindowEvidenceBatch,
    ) -> Iterator[tuple[np.ndarray, WindowReference]]:
        if isinstance(scores, WindowScoreBatch):
            for value, reference in zip(scores.values, scores.references, strict=True):
                yield np.full(reference.valid_length, value), reference
            return
        if isinstance(scores, WindowPointScoreBatch):
            yield from zip(scores.values, scores.references, strict=True)
            return
        raise TypeError("this propagation strategy does not use saliency evidence")

    @staticmethod
    def bounds(reference: WindowReference, series_length: int) -> tuple[int, int]:
        stop = reference.start + reference.valid_length
        if reference.start < 0 or stop > series_length:
            raise ValueError("window references must lie within the original series")
        return reference.start, stop

    def _resolve_uncovered(
        self, point_scores: np.ndarray, coverage: np.ndarray
    ) -> np.ndarray:
        missing = coverage == 0
        if not missing.any():
            return point_scores
        first_missing = int(np.flatnonzero(missing)[0])
        if (
            self.uncovered is not UncoveredPolicy.EDGE_PAD
            or first_missing == 0
            or not missing[first_missing:].all()
        ):
            raise ValueError("window configuration leaves uncovered time points")
        point_scores[first_missing:] = point_scores[first_missing - 1]
        return point_scores
