from types import MappingProxyType

import numpy as np
import pytest

from detectiv.models.autoencoders import TrainingHistory
from detectiv.reports import ReconstructionReport
from detectiv.typing import JSONValue


def _report(**metrics: float) -> ReconstructionReport:
    return ReconstructionReport(
        window_scores={},
        point_scores={"plan": {"mean": np.array([0.1, 0.9])}},
        training=TrainingHistory((0.5,)),
        metrics=metrics,
    )


def test_report_metrics_are_flat_immutable_and_used_consistently() -> None:
    report = _report(**{"plan.mean.ROC-AUC": 0.9})

    assert isinstance(report.metrics, MappingProxyType)
    assert report.record()["metrics"] == report.metrics
    assert report.summary().endswith("Metrics (plan / mean):\n  ROC-AUC: 0.900")
    with pytest.raises(TypeError):
        report.metrics["other"] = 0.5  # type: ignore[index]


def test_report_summary_compacts_only_a_useful_shared_metric_prefix() -> None:
    compact = _report(
        **{
            "plan.mean.ROC-AUC": 0.9,
            "plan.mean.VUS-PR": 0.8,
        }
    )
    unshared = _report(**{"train.loss": 0.5, "test.ROC-AUC": 0.9})

    assert compact.summary().endswith(
        "Metrics (plan / mean):\n  ROC-AUC: 0.900\n  VUS-PR: 0.800"
    )
    assert unshared.summary().endswith(
        "Metrics:\n  train.loss: 0.500\n  test.ROC-AUC: 0.900"
    )


def test_report_summary_handles_single_and_uneven_metric_names() -> None:
    single = _report(accuracy=0.75)
    uneven = _report(**{"plan.metric": 0.7, "plan.series.metric": 0.8})

    assert single.summary().endswith("Metrics:\n  accuracy: 0.750")
    assert uneven.summary().endswith(
        "Metrics (plan):\n  metric: 0.700\n  series.metric: 0.800"
    )


def test_reconstruction_summary_reports_image_splits_in_order() -> None:
    report = ReconstructionReport(
        window_scores={},
        point_scores={},
        training=TrainingHistory((0.5,)),
        resolved_inputs={
            "data": {
                "train": {"window_count": 12},
                "validation": {"window_count": 3},
                "test": {"window_count": 5},
            }
        },
    )

    assert report.summary().splitlines()[1:4] == [
        "Train images: 12",
        "Validation images: 3",
        "Test images: 5",
    ]


@pytest.mark.parametrize(
    "data",
    (
        None,
        [],
        {
            "train": {"window_count": "unknown"},
            "validation": None,
            "test": "unknown",
        },
    ),
)
def test_reconstruction_summary_handles_unavailable_image_metadata(
    data: JSONValue,
) -> None:
    report = ReconstructionReport(
        window_scores={},
        point_scores={},
        training=TrainingHistory((0.5,)),
        resolved_inputs={"data": data},
    )

    image_lines = [line for line in report.summary().splitlines() if "images:" in line]

    assert "Train images: unknown" not in image_lines
    assert "Test images: unknown" not in image_lines
    if isinstance(data, dict):
        assert image_lines == ["Validation images: none"]
    else:
        assert image_lines == []


def test_with_metrics_returns_an_immutable_replacement() -> None:
    original = _report()

    enriched = original.with_metrics({"plan.mean.metric": 1})

    assert enriched is not original
    assert original.metrics == {}
    assert enriched.metrics == {"plan.mean.metric": 1.0}


@pytest.mark.parametrize(
    ("metrics", "message"),
    (
        ({"": 0.5}, "names"),
        ({"metric": "0.5"}, "numeric"),
        ({"metric": True}, "numeric"),
        ({"metric": float("nan")}, "finite"),
        ({"metric": float("inf")}, "finite"),
    ),
)
def test_report_rejects_invalid_metrics(
    metrics: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ReconstructionReport(
            window_scores={},
            point_scores={},
            training=TrainingHistory((0.5,)),
            metrics=metrics,  # type: ignore[arg-type]
        )
