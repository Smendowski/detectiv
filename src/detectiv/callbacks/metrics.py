from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, Self

import numpy as np

from detectiv.callbacks.base import BaseCallback


class _Evaluator(Protocol):
    def evaluate(
        self, point_scores: np.ndarray, labels: np.ndarray
    ) -> Mapping[str, float]: ...


class _MetricsReport(Protocol):
    @property
    def point_scores(
        self,
    ) -> Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]: ...

    @property
    def point_labels(self) -> Mapping[str, np.ndarray] | None: ...

    def with_metrics(self, metrics: Mapping[str, float]) -> Self: ...


class MetricsCallback(BaseCallback[_MetricsReport]):
    """Evaluate every reconstructed point-score series after a successful run.

    Args:
        evaluator: Structural evaluator accepting aligned point scores and labels
            and returning named metrics.

    Labels are read from the reconstruction report, so callers do not supply
    them separately. All score branches are checked before evaluation begins.
    """

    def __init__(self, evaluator: _Evaluator) -> None:
        self.evaluator = evaluator

    @property
    def name(self) -> str:
        """Return the fixed registration name `metrics`.

        Returns:
            The callback registration name.
        """
        return "metrics"

    def on_run_finished[T: _MetricsReport](self, result: T) -> T:
        """Evaluate all point scores and return a metric-enriched report.

        Args:
            result: Completed reconstruction report containing scores and labels.

        Returns:
            An immutable replacement report containing qualified metric names.

        Raises:
            ValueError: If labels are absent or do not exactly cover and align
                with every scored series.
        """
        labels = result.point_labels
        if labels is None:
            raise ValueError(
                "metrics require point labels in the reconstruction report"
            )

        expected_series = set(labels)
        for plan, propagations in result.point_scores.items():
            for propagation, series_scores in propagations.items():
                if set(series_scores) != expected_series:
                    raise ValueError(
                        "point scores and labels must define the same series: "
                        f"{plan}.{propagation}"
                    )
                for series_id, point_scores in series_scores.items():
                    if point_scores.shape != labels[series_id].shape:
                        raise ValueError(
                            "point scores and labels must have the same shape: "
                            f"{plan}.{propagation}.{series_id}"
                        )

        metrics: dict[str, float] = {}
        for plan, propagations in result.point_scores.items():
            for propagation, series_scores in propagations.items():
                for series_id, point_scores in series_scores.items():
                    evaluated = self.evaluator.evaluate(point_scores, labels[series_id])
                    metrics.update(
                        {
                            f"{plan}.{propagation}.{series_id}.{metric}": value
                            for metric, value in evaluated.items()
                        }
                    )
        return result.with_metrics(metrics)
