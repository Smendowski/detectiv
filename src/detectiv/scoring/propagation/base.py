from abc import ABC, abstractmethod
from enum import StrEnum

import numpy as np

from detectiv.scoring.window_scores import (
    WindowEvidenceBatch,
    WindowPointScoreBatch,
)
from detectiv.time_series.windowing import WindowReference


class UncoveredPolicy(StrEnum):
    """How aggregation handles time points without window contributions."""

    ERROR = "error"
    EDGE_PAD = "edge_pad"


class PointAssignment(ABC):
    """Distribute each window's evidence over its valid source points."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable assignment identifier used in result keys."""
        raise NotImplementedError

    @abstractmethod
    def assign(self, evidence: WindowEvidenceBatch) -> WindowPointScoreBatch:
        """Assign every window's evidence to its valid source points."""
        raise NotImplementedError


class PointScoreAggregator(ABC):
    """Combine overlapping window contributions into one score per point."""

    def __init__(self, uncovered: UncoveredPolicy = UncoveredPolicy.ERROR) -> None:
        self.uncovered = uncovered

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable aggregation identifier used in result keys."""
        raise NotImplementedError

    def aggregate(
        self,
        contributions: WindowPointScoreBatch,
        series_length: int,
    ) -> np.ndarray:
        """Aggregate one series's contributions into a point-score array."""
        if series_length <= 0:
            raise ValueError("series_length must be positive")
        if not contributions.references:
            raise ValueError("at least one point contribution is required")
        if len({reference.series_id for reference in contributions.references}) != 1:
            raise ValueError("point contributions must belong to one series")

        point_scores, coverage = self._aggregate(contributions, series_length)
        return self._resolve_uncovered(point_scores, coverage)

    @abstractmethod
    def _aggregate(
        self,
        contributions: WindowPointScoreBatch,
        series_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def bounds(reference: WindowReference, series_length: int) -> tuple[int, int]:
        """Validate and return the valid source-point bounds of one window."""
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
        if self.uncovered is not UncoveredPolicy.EDGE_PAD:
            raise ValueError("window configuration leaves uncovered time points")
        if first_missing == 0 or not missing[first_missing:].all():
            raise ValueError("window configuration leaves uncovered time points")

        point_scores[first_missing:] = point_scores[first_missing - 1]
        return point_scores
