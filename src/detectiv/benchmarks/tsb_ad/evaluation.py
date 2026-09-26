from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from detectiv.utils import positive_integer

if TYPE_CHECKING:
    from detectiv.benchmarks.tsb_ad.adapter import TSBADAdapter


@dataclass(frozen=True)
class TSBADMetricEvaluator:
    """Evaluate pointwise anomaly scores with a configured TSB-AD metric.

    Args:
        adapter: Adapter that supplies the upstream TSB-AD metric function.
        sliding_window: Positive integral window length for metric adjustment.
        version: Upstream metric version. Defaults to ``"opt"``.
        thresholds: Positive integral number of thresholds to evaluate.
            Defaults to ``250``.

    Raises:
        ValueError: If ``sliding_window`` or ``thresholds`` is not a positive
            integer.
    """

    adapter: TSBADAdapter
    sliding_window: int
    version: str = "opt"
    thresholds: int = 250

    def __post_init__(self) -> None:
        sliding_window = positive_integer(self.sliding_window, "sliding_window")
        thresholds = positive_integer(self.thresholds, "thresholds")
        object.__setattr__(self, "sliding_window", sliding_window)
        object.__setattr__(self, "thresholds", thresholds)

    def evaluate(
        self, point_scores: np.ndarray, labels: np.ndarray
    ) -> dict[str, float]:
        """Evaluate pointwise anomaly scores against binary labels.

        Args:
            point_scores: One-dimensional finite anomaly scores.
            labels: One-dimensional binary labels containing both ``0`` and ``1``.

        Returns:
            Upstream TSB-AD metric names mapped to floating-point values.

        Raises:
            ValueError: If inputs are not one-dimensional with equal length,
                scores are empty or non-finite, or labels are not binary or do
                not contain both classes.
            ImportError: If the upstream package cannot be loaded.
            RuntimeError: If ``TSB_AD`` is already loaded from another source
                directory.
        """
        scores, validated_labels = _validate(point_scores, labels)

        return self.adapter._metrics(
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
        or not np.isin(labels, (0, 1)).all()
    ):
        raise ValueError("scores must be finite and labels must be binary")
    if labels.min() == labels.max():
        raise ValueError("labels must contain both normal and anomalous points")

    return scores, np.asarray(labels, dtype=np.int32)
