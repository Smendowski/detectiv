from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

import numpy as np

from detectiv.callbacks.base import ScenarioCallback

if TYPE_CHECKING:
    from detectiv.scenarios.reconstruction import ReconstructionScenarioResult


class _PointScoreEvaluator(Protocol):
    def evaluate(
        self, point_scores: np.ndarray, labels: np.ndarray
    ) -> Mapping[str, float]: ...


class EvaluationCallback(ScenarioCallback):
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
                                point_scores,
                                self.labels[series_id],
                            )
                        )
                    )
                    for series_id, point_scores in series_scores.items()
                }
                propagations[propagation] = MappingProxyType(series_metrics)
            plans[plan] = MappingProxyType(propagations)
        self.metrics = MappingProxyType(plans)


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
