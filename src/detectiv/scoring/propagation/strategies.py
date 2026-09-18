import numpy as np

from detectiv.scoring.propagation.base import PointAssignment, PointScoreAggregator
from detectiv.scoring.window_scores import (
    WindowEvidenceBatch,
    WindowPointScoreBatch,
    WindowScoreBatch,
)


class UniformPointAssignment(PointAssignment):
    @property
    def name(self) -> str:
        return "uniform"

    def assign(self, evidence: WindowEvidenceBatch) -> WindowPointScoreBatch:
        if not isinstance(evidence, WindowScoreBatch):
            raise TypeError("uniform assignment requires one score per window")
        return WindowPointScoreBatch(
            tuple(
                np.full(reference.valid_length, value)
                for value, reference in zip(
                    evidence.values, evidence.references, strict=True
                )
            ),
            evidence.references,
        )


class MeanPointScoreAggregator(PointScoreAggregator):
    @property
    def name(self) -> str:
        return "mean"

    def _aggregate(
        self,
        contributions: WindowPointScoreBatch,
        series_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        point_scores = np.zeros(series_length, dtype=np.float64)
        coverage = np.zeros(series_length, dtype=np.intp)
        for values, reference in zip(
            contributions.values, contributions.references, strict=True
        ):
            start, stop = self.bounds(reference, series_length)
            point_scores[start:stop] += values
            coverage[start:stop] += 1
        return (
            np.divide(
                point_scores,
                coverage,
                out=np.zeros(series_length),
                where=coverage > 0,
            ),
            coverage,
        )


class MaxPointScoreAggregator(PointScoreAggregator):
    @property
    def name(self) -> str:
        return "max"

    def _aggregate(
        self,
        contributions: WindowPointScoreBatch,
        series_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        point_scores = np.full(series_length, -np.inf)
        coverage = np.zeros(series_length, dtype=np.intp)
        for values, reference in zip(
            contributions.values, contributions.references, strict=True
        ):
            start, stop = self.bounds(reference, series_length)
            point_scores[start:stop] = np.maximum(point_scores[start:stop], values)
            coverage[start:stop] += 1
        return point_scores, coverage


class MedianPointScoreAggregator(PointScoreAggregator):
    @property
    def name(self) -> str:
        return "median"

    def _aggregate(
        self,
        contributions: WindowPointScoreBatch,
        series_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        point_contributions: list[list[float]] = [[] for _ in range(series_length)]
        for values, reference in zip(
            contributions.values, contributions.references, strict=True
        ):
            start, _ = self.bounds(reference, series_length)
            for point, value in enumerate(values, start=start):
                point_contributions[point].append(float(value))

        coverage = np.asarray(
            [len(values) for values in point_contributions], dtype=np.intp
        )
        point_scores = np.asarray(
            [np.median(values) if values else 0.0 for values in point_contributions],
            dtype=np.float64,
        )
        return point_scores, coverage
