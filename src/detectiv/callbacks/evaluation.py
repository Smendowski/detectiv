from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import numpy as np

from detectiv.callbacks.base import BaseCallback


class _PointScoreResult(Protocol):
    @property
    def point_scores(self) -> Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]: ...


class _PointScoreEvaluator(Protocol):
    def evaluate(
        self, point_scores: np.ndarray, labels: np.ndarray
    ) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class EvaluationRecord:
    """Named metrics for one series under one scoring plan and propagation."""

    plan: str
    propagation: str
    series_id: str
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class EvaluationReport:
    """Immutable collection of per-series evaluation records."""

    records: tuple[EvaluationRecord, ...]

    def as_dict(self) -> Mapping[str, object]:
        """Return a JSON-ready representation preserving every evaluation record."""
        return MappingProxyType(
            {
                "records": [
                    {
                        "plan": record.plan,
                        "propagation": record.propagation,
                        "series_id": record.series_id,
                        "metrics": dict(record.metrics),
                    }
                    for record in self.records
                ]
            }
        )

    def tracking_metrics(self) -> Mapping[str, Mapping[str, Mapping[str, float]]]:
        """Return immutable mean metrics grouped by plan and propagation."""
        plans: dict[str, dict[str, dict[str, list[float]]]] = {}
        for record in self.records:
            values = plans.setdefault(record.plan, {}).setdefault(
                record.propagation, {}
            )
            for name, value in record.metrics.items():
                values.setdefault(name, []).append(value)
        return MappingProxyType(
            {
                plan: MappingProxyType(
                    {
                        propagation: MappingProxyType(
                            {
                                name: float(np.mean(values))
                                for name, values in metric_values.items()
                            }
                        )
                        for propagation, metric_values in propagations.items()
                    }
                )
                for plan, propagations in plans.items()
            }
        )


class EvaluationCallback[T: _PointScoreResult](BaseCallback[T]):
    """Evaluate propagated point scores against immutable binary labels.

    Args:
        evaluator: Object whose `evaluate(point_scores, labels)` returns named
            numeric metrics for one series.
        labels: Non-empty, one-dimensional arrays of only zero and one, keyed
            by series ID.

    Labels are copied and made read-only. Successful completion stores an
    immutable report of records by scoring plan, propagation, and series.
    """

    def __init__(
        self,
        evaluator: _PointScoreEvaluator,
        labels: Mapping[str, np.ndarray],
    ) -> None:
        """Validate and freeze labels for use at successful completion.

        Raises:
            ValueError: If any label key is empty or its values are not a
                non-empty one-dimensional binary array.
        """
        self.evaluator = evaluator
        self.labels = _validated_labels(labels)
        self.metrics: EvaluationReport | None = None

    @property
    def name(self) -> str:
        """Return the fixed registration name `evaluation`."""
        return "evaluation"

    def on_run_started(self) -> None:
        """Clear metrics from any preceding run."""
        self.metrics = None

    def on_run_finished(self, result: T) -> None:
        """Evaluate every result point-score array against its series labels.

        Each propagation must define exactly the configured label series. Point
        scores must be finite, one-dimensional, and length-aligned with labels.
        The evaluator's mappings are copied into immutable metric mappings.

        Raises:
            ValueError: If result series, score shape, score finiteness, or
                score length violates these requirements.
        """
        if any(
            set(series_scores) != set(self.labels)
            for scores in result.point_scores.values()
            for series_scores in scores.values()
        ):
            raise ValueError("point scores and labels must define the same series")

        records: list[EvaluationRecord] = []
        for plan_name, propagations in result.point_scores.items():
            for propagation_name, scores_by_series in propagations.items():
                for series_id, point_scores in scores_by_series.items():
                    labels = self.labels[series_id]
                    records.append(
                        EvaluationRecord(
                            plan=plan_name,
                            propagation=propagation_name,
                            series_id=series_id,
                            metrics=MappingProxyType(
                                dict(
                                    self.evaluator.evaluate(
                                        _validated_point_scores(point_scores, labels),
                                        labels,
                                    )
                                )
                            ),
                        )
                    )
        self.metrics = EvaluationReport(tuple(records))

    def tracking_metrics(self) -> Mapping[str, Mapping[str, Mapping[str, float]]]:
        """Return immutable per-plan, per-propagation mean series metrics.

        Returns:
            Metrics averaged independently across series for each plan and
            propagation.

        Raises:
            RuntimeError: If no successful completion has produced metrics.
        """
        if self.metrics is None:
            raise RuntimeError("evaluation metrics are not available")
        return self.metrics.tracking_metrics()


def _validated_labels(labels: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for series_id, labels_for_series in labels.items():
        values_for_series = np.asarray(labels_for_series)
        if (
            not series_id
            or values_for_series.ndim != 1
            or not len(values_for_series)
            or not np.isin(values_for_series, (0, 1)).all()
        ):
            raise ValueError("labels must be non-empty one-dimensional binary arrays")
        values_for_series = np.array(values_for_series, dtype=np.int32, copy=True)
        values_for_series.setflags(write=False)
        values[series_id] = values_for_series
    return MappingProxyType(values)


def _validated_point_scores(point_scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(point_scores, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(labels) or not np.isfinite(values).all():
        raise ValueError(
            "point scores must be finite one-dimensional arrays aligned with labels"
        )
    return values
