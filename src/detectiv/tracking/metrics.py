from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from detectiv.evaluation import PointScoreEvaluator
from detectiv.scenarios.callbacks import ScenarioCallback
from detectiv.scenarios.reconstruction import ReconstructionScenarioResult


class MetricsCallback(ScenarioCallback):
    def __init__(
        self,
        evaluator: PointScoreEvaluator,
        labels: Mapping[str, np.ndarray],
    ) -> None:
        self.evaluator = evaluator
        self.labels = _validated_labels(labels)
        self.metrics: (
            Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]] | None
        ) = None

    def on_run_started(self) -> None:
        self.metrics = None

    def on_run_finished(self, result: ReconstructionScenarioResult) -> None:
        if any(
            set(series_scores) != set(self.labels)
            for scores in result.point_scores.values()
            for series_scores in scores.values()
        ):
            raise ValueError("point scores and labels must define the same series")
        self.metrics = MappingProxyType(
            {
                plan: MappingProxyType(
                    {
                        propagation: MappingProxyType(
                            dict(
                                (
                                    series_id,
                                    MappingProxyType(
                                        dict(
                                            self.evaluator.evaluate(
                                                series_scores[series_id],
                                                self.labels[series_id],
                                            )
                                        )
                                    ),
                                )
                                for series_id in series_scores
                            )
                        )
                        for propagation, series_scores in scores.items()
                    }
                )
                for plan, scores in result.point_scores.items()
            }
        )


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
