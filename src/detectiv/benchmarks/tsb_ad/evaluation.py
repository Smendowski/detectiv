from dataclasses import dataclass
from operator import index

import numpy as np

from detectiv.benchmarks.tsb_ad.adapter import TSBADAdapter


@dataclass(frozen=True)
class TSBADEvaluator:
    """Evaluate pointwise anomaly scores with a configured TSB-AD metric.

    Args:
        repository: Adapter that supplies the upstream TSB-AD metric function.
        sliding_window: Positive integral window length for metric adjustment.
        version: Upstream metric version. Defaults to ``"opt"``.
        thresholds: Positive integral number of thresholds to evaluate.
            Defaults to ``250``.

    Raises:
        ValueError: If ``sliding_window`` or ``thresholds`` is not a positive
            integer.
    """

    repository: TSBADAdapter
    sliding_window: int
    version: str = "opt"
    thresholds: int = 250

    def __post_init__(self) -> None:
        sliding_window = _positive_index(self.sliding_window, "sliding_window")
        thresholds = _positive_index(self.thresholds, "thresholds")
        object.__setattr__(self, "sliding_window", sliding_window)
        object.__setattr__(self, "thresholds", thresholds)

    def evaluate(
        self, point_scores: np.ndarray, labels: np.ndarray
    ) -> dict[str, float]:
        """Evaluate pointwise anomaly scores against binary labels.

        Args:
            point_scores: One-dimensional finite anomaly scores.
            labels: One-dimensional boolean or integer labels containing both
                ``0`` and ``1``.

        Returns:
            Upstream TSB-AD metric names mapped to floating-point values.

        Raises:
            ValueError: If inputs are not one-dimensional with equal length,
                scores are empty or non-finite, or labels are not binary
                booleans or integers containing both classes.
            ImportError: If the upstream package cannot be loaded.
            RuntimeError: If ``TSB_AD`` is already loaded from another source
                directory.
        """
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
    if not np.issubdtype(labels.dtype, np.bool_) and not np.issubdtype(
        labels.dtype, np.integer
    ):
        raise ValueError("labels must be boolean or integer binary values")
    if (
        not len(scores)
        or not np.isfinite(scores).all()
        or not np.isin(labels, (0, 1)).all()
    ):
        raise ValueError("scores must be finite and labels must be binary")
    if labels.min() == labels.max():
        raise ValueError("labels must contain both normal and anomalous points")

    return scores, np.asarray(labels, dtype=np.int32)


def _positive_index(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        value = index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
