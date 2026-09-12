import numpy as np

from detectiv.scoring.propagation.base import WindowToPointScorePropagationStrategy
from detectiv.scoring.window_scores import (
    WindowEvidenceBatch,
    WindowPointScoreBatch,
    WindowSaliencyBatch,
)


class MeanPropagationStrategy(WindowToPointScorePropagationStrategy):
    @property
    def name(self) -> str:
        return "mean"

    def _aggregate(
        self,
        scores: WindowEvidenceBatch,
        series_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        point_scores = np.zeros(series_length, dtype=np.float64)
        coverage = np.zeros(series_length, dtype=np.intp)
        for values, reference in self.contributions(scores):
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


class MaxPropagationStrategy(WindowToPointScorePropagationStrategy):
    @property
    def name(self) -> str:
        return "max"

    def _aggregate(
        self,
        scores: WindowEvidenceBatch,
        series_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        point_scores = np.full(series_length, -np.inf)
        coverage = np.zeros(series_length, dtype=np.intp)
        for values, reference in self.contributions(scores):
            start, stop = self.bounds(reference, series_length)
            point_scores[start:stop] = np.maximum(point_scores[start:stop], values)
            coverage[start:stop] += 1
        return point_scores, coverage


class MedianPropagationStrategy(WindowToPointScorePropagationStrategy):
    @property
    def name(self) -> str:
        return "median"

    def _aggregate(
        self,
        scores: WindowEvidenceBatch,
        series_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        contributions: list[list[float]] = [[] for _ in range(series_length)]
        for values, reference in self.contributions(scores):
            start, _ = self.bounds(reference, series_length)
            for point, value in enumerate(values, start=start):
                contributions[point].append(float(value))
        coverage = np.asarray([len(values) for values in contributions], dtype=np.intp)
        point_scores = np.asarray(
            [np.median(values) if values else 0.0 for values in contributions],
            dtype=np.float64,
        )
        return point_scores, coverage


class TemporalColumnPropagationStrategy(MeanPropagationStrategy):
    @property
    def name(self) -> str:
        return "temporal_column"

    def _aggregate(
        self,
        scores: WindowEvidenceBatch,
        series_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(scores, WindowPointScoreBatch):
            raise TypeError(
                "temporal-column propagation requires time-resolved window scores"
            )
        return super()._aggregate(scores, series_length)


class SaliencyWeightedPropagationStrategy(MeanPropagationStrategy):
    @property
    def name(self) -> str:
        return "saliency_weighted"

    def _aggregate(
        self,
        scores: WindowEvidenceBatch,
        series_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(scores, WindowSaliencyBatch):
            raise TypeError(
                "saliency-weighted propagation requires window scores and saliencies"
            )

        weighted_scores = tuple(
            score * saliency / saliency.mean()
            for score, saliency in zip(scores.values, scores.saliencies, strict=True)
        )
        return super()._aggregate(
            WindowPointScoreBatch(weighted_scores, scores.references), series_length
        )
