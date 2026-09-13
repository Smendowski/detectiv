from dataclasses import dataclass

import numpy as np

from detectiv.benchmarks.tsb_ad.repository import TSBADRepository


@dataclass(frozen=True)
class TSBADEvaluator:
    repository: TSBADRepository
    sliding_window: int
    version: str = "opt"
    thresholds: int = 250

    def __post_init__(self) -> None:
        if self.sliding_window <= 0 or self.thresholds <= 0:
            raise ValueError("sliding_window and thresholds must be positive")

    def evaluate(
        self, point_scores: np.ndarray, labels: np.ndarray
    ) -> dict[str, float]:
        scores, validated_labels = _validate(point_scores, labels)
        return self.repository._metrics(
            scores,
            validated_labels,
            sliding_window=self.sliding_window,
            version=self.version,
            thresholds=self.thresholds,
        )


def _validate(
    point_scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(point_scores, dtype=np.float64)
    labels = np.asarray(labels)
    if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels):
        raise ValueError(
            "point_scores and labels must be one-dimensional with equal length"
        )
    if (
        not len(scores)
        or not np.isfinite(scores).all()
        or not np.issubdtype(labels.dtype, np.number)
        or not np.isfinite(labels).all()
        or not np.isin(labels, (0, 1)).all()
    ):
        raise ValueError("scores must be finite and labels must be binary")
    if labels.min() == labels.max():
        raise ValueError("labels must contain both normal and anomalous points")
    return scores, np.asarray(labels, dtype=np.int32)
