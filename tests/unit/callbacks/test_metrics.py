from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from detectiv.callbacks import MetricsCallback
from detectiv.models.autoencoders import TrainingHistory
from detectiv.scenarios import ReconstructionReport


class FakeEvaluator:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def evaluate(
        self, point_scores: np.ndarray, labels: np.ndarray
    ) -> Mapping[str, float]:
        self.calls.append((point_scores, labels))
        return {"score": float(point_scores.mean())}


def test_metrics_callback_evaluates_every_score_series_and_persists_metrics(
    tmp_path: Path,
) -> None:
    evaluator = FakeEvaluator()
    labels = {
        "first": np.array([0, 1]),
        "second": np.array([1, 0]),
    }
    point_scores = {
        "plan_a": {
            "mean": {
                "first": np.array([0.1, 0.9]),
                "second": np.array([0.8, 0.2]),
            },
            "max": {
                "first": np.array([0.2, 1.0]),
                "second": np.array([0.7, 0.1]),
            },
        },
        "plan_b": {
            "mean": {
                "first": np.array([0.3, 0.7]),
                "second": np.array([0.6, 0.4]),
            }
        },
    }
    report = _report(point_scores=point_scores, point_labels=labels)
    result = MetricsCallback(evaluator).on_run_finished(report)

    assert result.metrics == pytest.approx(
        {
            "plan_a.mean.first.score": 0.5,
            "plan_a.mean.second.score": 0.5,
            "plan_a.max.first.score": 0.6,
            "plan_a.max.second.score": 0.4,
            "plan_b.mean.first.score": 0.5,
            "plan_b.mean.second.score": 0.5,
        }
    )
    assert len(evaluator.calls) == 6
    assert report.point_labels is not None
    assert evaluator.calls[0][1] is report.point_labels["first"]
    artifacts = result.write(tmp_path / "run")
    assert artifacts.read_report()["metrics"] == result.metrics


@pytest.mark.parametrize(
    "point_scores",
    (
        {
            "valid": {"mean": {"series": np.array([0.1, 0.9])}},
            "invalid": {"mean": {"other": np.array([0.1, 0.9])}},
        },
        {
            "valid": {"mean": {"series": np.array([0.1, 0.9])}},
            "invalid": {"mean": {"series": np.array([0.1])}},
        },
    ),
)
def test_metrics_callback_validates_every_branch_before_evaluation(
    point_scores: dict[str, dict[str, dict[str, np.ndarray]]],
) -> None:
    evaluator = FakeEvaluator()

    with pytest.raises(ValueError, match="point scores and labels"):
        MetricsCallback(evaluator).on_run_finished(
            _report(
                point_scores=point_scores,
                point_labels={"series": np.array([0, 1])},
            )
        )

    assert evaluator.calls == []


def test_metrics_callback_requires_report_labels() -> None:
    evaluator = FakeEvaluator()

    with pytest.raises(ValueError, match="require point labels"):
        MetricsCallback(evaluator).on_run_finished(
            _report(
                point_scores={"plan": {"mean": {"series": np.array([0.1, 0.9])}}},
                point_labels=None,
            )
        )

    assert evaluator.calls == []


def _report(
    *,
    point_scores: dict[str, dict[str, dict[str, np.ndarray]]],
    point_labels: dict[str, np.ndarray] | None,
) -> ReconstructionReport:
    return ReconstructionReport(
        window_scores={},
        point_scores=point_scores,
        point_labels=point_labels,
        training=TrainingHistory((0.5,)),
    )
