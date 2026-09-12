import numpy as np

from detectiv.data import WindowReference
from detectiv.evaluation import PointScoreEvaluator
from detectiv.models.autoencoders import TrainingHistory
from detectiv.scenarios.reconstruction import ReconstructionScenarioResult
from detectiv.scoring import WindowScoreBatch
from detectiv.tracking import MetricsCallback


class MeanScoreEvaluator(PointScoreEvaluator):
    def evaluate(
        self, point_scores: np.ndarray, labels: np.ndarray
    ) -> dict[str, float]:
        assert point_scores.shape == labels.shape
        return {"mean": float(point_scores.mean())}


def test_metrics_callback_evaluates_each_original_label_series() -> None:
    callback = MetricsCallback(
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

    assert callback.metrics == {"window": {"mean": {"series": {"mean": 2.0}}}}
