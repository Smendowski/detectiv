from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

import numpy as np

from detectiv.callbacks.base import BaseCallback

if TYPE_CHECKING:
    from detectiv.scenarios.reconstruction import ReconstructionScenarioResult


class _PointScoreEvaluator(Protocol):
    def evaluate(
        self, point_scores: np.ndarray, labels: np.ndarray
    ) -> Mapping[str, float]: ...


class EvaluationCallback(BaseCallback):
    def __init__(
        self,
        evaluator: _PointScoreEvaluator,
        labels: Mapping[str, np.ndarray],
    ) -> None:
        self.evaluator = evaluator
        self.labels = _validated_labels(labels)
        self.metrics: (
            Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]] | None
        ) = None

    @property
    def name(self) -> str:
        return "evaluation"

    def on_run_started(self) -> None:
        self.metrics = None

    def on_run_finished(self, result: "ReconstructionScenarioResult") -> None:
        if any(
            set(series_scores) != set(self.labels)
            for scores in result.point_scores.values()
            for series_scores in scores.values()
        ):
            raise ValueError("point scores and labels must define the same series")

        plans: dict[str, Mapping[str, Mapping[str, Mapping[str, float]]]] = {}
        for plan, scores in result.point_scores.items():
            propagations: dict[str, Mapping[str, Mapping[str, float]]] = {}
            for propagation, series_scores in scores.items():
                series_metrics = {
                    series_id: MappingProxyType(
                        dict(
                            self.evaluator.evaluate(
                                _validated_point_scores(
                                    point_scores,
                                    self.labels[series_id],
                                ),
                                self.labels[series_id],
                            )
                        )
                    )
                    for series_id, point_scores in series_scores.items()
                }
                propagations[propagation] = MappingProxyType(series_metrics)
            plans[plan] = MappingProxyType(propagations)
        self.metrics = MappingProxyType(plans)

    def tracking_metrics(self) -> Mapping[str, Mapping[str, Mapping[str, float]]]:
        if self.metrics is None:
            raise RuntimeError("evaluation metrics are not available")
        plans: dict[str, Mapping[str, Mapping[str, float]]] = {}
        for plan, propagations in self.metrics.items():
            aggregated: dict[str, Mapping[str, float]] = {}
            for propagation, series_metrics in propagations.items():
                values: dict[str, list[float]] = {}
                for metrics in series_metrics.values():
                    for name, value in metrics.items():
                        values.setdefault(name, []).append(value)
                aggregated[propagation] = MappingProxyType(
                    {
                        name: float(np.mean(metric_values))
                        for name, metric_values in values.items()
                    }
                )
            plans[plan] = MappingProxyType(aggregated)
        return MappingProxyType(plans)


def _validated_labels(labels: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for series_id, labels_for_series in labels.items():
        values_for_series = np.asarray(labels_for_series, dtype=np.int32)
        if (
            not series_id
            or values_for_series.ndim != 1
            or not len(values_for_series)
            or not np.isin(values_for_series, (0, 1)).all()
        ):
            raise ValueError("labels must be non-empty one-dimensional binary arrays")
        values_for_series = np.array(values_for_series, copy=True)
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
