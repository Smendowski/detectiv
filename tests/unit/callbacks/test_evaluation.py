import numpy as np
import pytest

from detectiv.callbacks import (
    EvaluationCallback,
    EvaluationRecord,
    EvaluationReport,
)
from detectiv.models.autoencoders import TrainingHistory
from detectiv.scenarios import ReconstructionScenarioResult
from detectiv.scoring import WindowScoreBatch
from detectiv.time_series.windowing import WindowReference


class MeanScoreEvaluator:
    def evaluate(
        self, point_scores: np.ndarray, labels: np.ndarray
    ) -> dict[str, float]:
        assert point_scores.shape == labels.shape
        return {"mean": float(point_scores.mean())}


def test_metrics_callback_evaluates_each_original_label_series() -> None:
    callback: EvaluationCallback[ReconstructionScenarioResult] = EvaluationCallback(
        MeanScoreEvaluator(),
        {"series": np.array([0, 1, 0], dtype=np.int32)},
    )
    result = ReconstructionScenarioResult(
        window_scores={
            "window": WindowScoreBatch(
                np.array([1.0]), (WindowReference("series", 0, 3, 3),)
            )
        },
        point_scores={"window": {"mean": {"series": np.array([1.0, 2.0, 3.0])}}},
        training=TrainingHistory((0.5,)),
    )

    callback.on_run_started()
    callback.on_run_finished(result)

    assert callback.metrics == EvaluationReport(
        (EvaluationRecord("window", "mean", "series", {"mean": 2.0}),)
    )
    assert callback.tracking_metrics() == {"window": {"mean": {"mean": 2.0}}}


def test_metrics_callback_rejects_unaligned_point_scores() -> None:
    callback: EvaluationCallback[ReconstructionScenarioResult] = EvaluationCallback(
        MeanScoreEvaluator(), {"series": np.array([0, 1])}
    )
    result = ReconstructionScenarioResult(
        window_scores={},
        point_scores={"plan": {"mean": {"series": np.array([1.0])}}},
        training=TrainingHistory((0.5,)),
    )

    with pytest.raises(ValueError, match="aligned with labels"):
        callback.on_run_finished(result)


@pytest.mark.parametrize("labels", [np.array([0.5, 1.0]), np.array([0.0, 1.9])])
def test_metrics_callback_rejects_fractional_labels(labels: np.ndarray) -> None:
    with pytest.raises(ValueError, match="binary"):
        EvaluationCallback(MeanScoreEvaluator(), {"series": labels})
