from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from detectiv.callbacks import MetricsCallback
from detectiv.models.autoencoders import TrainingHistory
from detectiv.reports import ReconstructionReport


class FakeEvaluator:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def evaluate(
        self, point_scores: np.ndarray, labels: np.ndarray
    ) -> Mapping[str, float]:
        self.calls.append((point_scores, labels))
        return {"score": float(point_scores.mean())}


def test_metrics_callback_evaluates_every_score_branch_and_persists_metrics(
    tmp_path: Path,
) -> None:
    evaluator = FakeEvaluator()
    labels = np.array([0, 1])
    point_scores = {
        "plan_a": {
            "mean": np.array([0.1, 0.9]),
            "max": np.array([0.2, 1.0]),
        },
        "plan_b": {"mean": np.array([0.3, 0.7])},
    }
    report = _report(point_scores=point_scores, point_labels=labels)
    result = MetricsCallback(evaluator).on_run_finished(report)

    assert result.metrics == pytest.approx(
        {
            "plan_a.mean.score": 0.5,
            "plan_a.max.score": 0.6,
            "plan_b.mean.score": 0.5,
        }
    )
    assert len(evaluator.calls) == 3
    assert report.point_labels is not None
    assert evaluator.calls[0][1] is report.point_labels
    artifacts = result.write(tmp_path / "run")
    assert artifacts.read_report()["metrics"] == result.metrics


@pytest.mark.parametrize(
    "point_scores",
    ({"valid": {"mean": np.array([0.1, 0.9])}, "invalid": {"mean": np.array([0.1])}},),
)
def test_metrics_callback_validates_every_branch_before_evaluation(
    point_scores: dict[str, dict[str, np.ndarray]],
) -> None:
    evaluator = FakeEvaluator()

    with pytest.raises(ValueError, match="point scores and labels"):
        MetricsCallback(evaluator).on_run_finished(
            _report(
                point_scores=point_scores,
                point_labels=np.array([0, 1]),
            )
        )

    assert evaluator.calls == []


def test_metrics_callback_requires_report_labels() -> None:
    evaluator = FakeEvaluator()

    with pytest.raises(ValueError, match="require point labels"):
        MetricsCallback(evaluator).on_run_finished(
            _report(
                point_scores={"plan": {"mean": np.array([0.1, 0.9])}},
                point_labels=None,
            )
        )

    assert evaluator.calls == []


def _report(
    *,
    point_scores: dict[str, dict[str, np.ndarray]],
    point_labels: np.ndarray | None,
) -> ReconstructionReport:
    return ReconstructionReport(
        window_scores={},
        point_scores=point_scores,
        point_labels=point_labels,
        training=TrainingHistory((0.5,)),
    )
