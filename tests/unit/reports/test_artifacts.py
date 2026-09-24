import hashlib
import json
import re
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from detectiv.models.autoencoders import TrainingHistory
from detectiv.reports import (
    ReconstructionReport,
    ReconstructionReportWriter,
    ReportArtifacts,
)
from detectiv.runs import JSONValue
from detectiv.scoring import WindowScoreBatch
from detectiv.time_series.windowing import WindowReference


def test_report_writer_separates_manifest_and_point_scores(
    tmp_path: Path,
) -> None:
    result = ReconstructionReport(
        window_scores={
            "window": WindowScoreBatch(
                np.array([1.0]), (WindowReference("series", 0, 3, 3),)
            )
        },
        point_scores={"window": {"mean": np.array([1.0, 2.0, 3.0])}},
        training=TrainingHistory((0.5,)),
        metrics={"window.mean.AUC-PR": 0.8},
    )

    artifacts = ReconstructionReportWriter(
        tmp_path, provenance={"dataset": {"name": "synthetic"}}
    ).write(result)

    manifest = json.loads(artifacts.manifest.read_text())
    scores = np.load(artifacts.point_scores)
    assert manifest["schema_version"] == 3
    assert manifest["scenario_type"] == "reconstruction"
    assert manifest["training"] == {
        "losses": [0.5],
        "validation_losses": [],
        "best_epoch": None,
        "best_validation_loss": None,
        "device": None,
    }
    assert manifest["provenance"] == {"dataset": {"name": "synthetic"}}
    assert manifest["reproducibility"] == {}
    assert manifest["resolved_inputs"] == {}
    assert "inputs" not in manifest
    assert manifest["runtime"]["source_revision"]
    assert manifest["runtime"]["dependency_lock_sha256"]
    assert manifest["scores"]["keys"] == {"window": {"mean": "score_0_0"}}
    assert manifest["metrics"] == {"window.mean.AUC-PR": 0.8}
    assert (
        manifest["artifacts"]["point_scores.npz"]
        == hashlib.file_digest(artifacts.point_scores.open("rb"), "sha256").hexdigest()
    )
    np.testing.assert_allclose(scores["score_0_0"], [1.0, 2.0, 3.0])


def test_reconstruction_report_writes_its_metrics(tmp_path: Path) -> None:
    report = ReconstructionReport(
        window_scores={},
        point_scores={"window": {"mean": np.array([1.0])}},
        training=TrainingHistory((0.5,)),
        metrics={"window.mean.ROC-AUC": 0.9},
    )

    artifacts = report.write(tmp_path)

    assert report.summary().endswith("Metrics (window / mean):\n  ROC-AUC: 0.900")
    assert tuple(report.metrics) == ("window.mean.ROC-AUC",)
    assert artifacts.read_report()["metrics"] == {"window.mean.ROC-AUC": 0.9}


def test_report_writer_creates_opt_in_figures(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib.pyplot")
    result = ReconstructionReport(
        window_scores={},
        point_scores={"window": {"mean": np.array([1.0, 2.0])}},
        training=TrainingHistory((0.5,), validation_losses=(0.4,)),
    )

    artifacts = ReconstructionReportWriter(tmp_path, visualize=True).write(result)

    assert [path.relative_to(tmp_path).as_posix() for path in artifacts.figures] == [
        "figures/training_loss.png",
        "figures/score_0_0.png",
    ]


def test_report_artifacts_open_verify_and_load_scores(tmp_path: Path) -> None:
    result = ReconstructionReport(
        window_scores={},
        point_scores={"window": {"mean": np.array([1.0, 2.0])}},
        training=TrainingHistory((0.5,)),
    )
    ReconstructionReportWriter(tmp_path).write(result)

    artifacts = ReportArtifacts.open(tmp_path)

    artifacts.verify()
    scores = artifacts.load_scores()
    np.testing.assert_allclose(scores["score_0_0"], [1.0, 2.0])
    assert not scores["score_0_0"].flags.writeable


def test_report_artifacts_verified_access_checks_before_reading(tmp_path: Path) -> None:
    result = ReconstructionReport(
        window_scores={},
        point_scores={"window": {"mean": np.array([1.0])}},
        training=TrainingHistory((0.5,)),
    )
    artifacts = ReconstructionReportWriter(tmp_path).write(result)
    artifacts.point_scores.write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="checksum"):
        ReportArtifacts.open_verified(tmp_path)


def test_report_artifacts_detect_corrupted_scores(tmp_path: Path) -> None:
    result = ReconstructionReport(
        window_scores={},
        point_scores={"window": {"mean": np.array([1.0])}},
        training=TrainingHistory((0.5,)),
    )
    artifacts = ReconstructionReportWriter(tmp_path).write(result)
    artifacts.point_scores.write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="checksum"):
        ReportArtifacts.open(tmp_path).verify()


def test_report_artifacts_reject_missing_score_checksum(tmp_path: Path) -> None:
    result = ReconstructionReport(
        window_scores={},
        point_scores={"window": {"mean": np.array([1.0])}},
        training=TrainingHistory((0.5,)),
    )
    artifacts = ReconstructionReportWriter(tmp_path).write(result)
    report = json.loads(artifacts.manifest.read_text())
    del report["artifacts"]["point_scores.npz"]
    artifacts.manifest.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        ReportArtifacts.open(tmp_path)


def test_report_writer_refuses_non_empty_destination(tmp_path: Path) -> None:
    destination = tmp_path / "run"
    destination.mkdir()
    marker = destination / "existing.txt"
    marker.write_text("keep", encoding="utf-8")
    result = ReconstructionReport({}, {}, TrainingHistory((0.5,)))

    with pytest.raises(FileExistsError, match="overwrite=True"):
        ReconstructionReportWriter(destination).write(result)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_report_writer_replaces_only_when_requested(tmp_path: Path) -> None:
    destination = tmp_path / "run"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")
    result = ReconstructionReport({}, {}, TrainingHistory((0.5,)))

    artifacts = ReconstructionReportWriter(destination, overwrite=True).write(result)

    assert artifacts.manifest.is_file()
    assert not (destination / "old.txt").exists()


def test_report_writer_does_not_publish_interrupted_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "run"
    result = ReconstructionReport({}, {}, TrainingHistory((0.5,)))

    def interrupt(*args: object, **kwargs: object) -> ReportArtifacts:
        raise KeyboardInterrupt

    monkeypatch.setattr(ReconstructionReportWriter, "_write", interrupt)

    with pytest.raises(KeyboardInterrupt):
        ReconstructionReportWriter(destination).write(result)

    assert not destination.exists()


def test_artifact_writer_preserves_nested_json_metadata(tmp_path: Path) -> None:
    result = ReconstructionReport(
        window_scores={},
        point_scores={},
        training=TrainingHistory((0.5,)),
        reproducibility={"runtime": {"devices": ["cpu"]}},
        resolved_inputs={"data": {"partitions": ["train", "test"]}},
    )

    artifacts = ReconstructionReportWriter(
        tmp_path, provenance={"source": {"files": ["data.csv"]}}
    ).write(result)

    report = json.loads(artifacts.manifest.read_text())
    assert report["provenance"]["source"]["files"] == ["data.csv"]


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_artifact_writer_rejects_non_finite_metadata(
    tmp_path: Path, value: float
) -> None:
    result = ReconstructionReport(
        window_scores={},
        point_scores={},
        training=TrainingHistory((0.5,)),
        reproducibility={"seed": value},
        resolved_inputs={},
    )

    with pytest.raises(ValueError, match=r"reproducibility\.seed must be finite"):
        ReconstructionReportWriter(tmp_path).write(result)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_artifact_writer_rejects_non_finite_scores(
    tmp_path: Path, value: float
) -> None:
    result = ReconstructionReport(
        window_scores={},
        point_scores={"window": {"mean": np.array([value])}},
        training=TrainingHistory((0.5,)),
    )

    with pytest.raises(ValueError, match="point scores must contain"):
        ReconstructionReportWriter(tmp_path).write(result)


def test_json_conversion_rejects_invalid_values_explicitly() -> None:
    from detectiv.reports.artifacts import _json_value

    with pytest.raises(ValueError, match="not compatible"):
        _json_value(object())


@pytest.mark.parametrize(
    ("provenance", "reproducibility", "resolved_inputs", "path"),
    (
        ({"source": cast(JSONValue, object())}, {}, {}, "provenance.source"),
        ({}, {"runtime": cast(JSONValue, object())}, {}, "reproducibility.runtime"),
        (
            {},
            {},
            {"data": [cast(JSONValue, object())]},
            "resolved_inputs.data[0]",
        ),
    ),
)
def test_artifact_writer_rejects_invalid_metadata_before_publication(
    tmp_path: Path,
    provenance: dict[str, JSONValue],
    reproducibility: dict[str, JSONValue],
    resolved_inputs: dict[str, JSONValue],
    path: str,
) -> None:
    result = ReconstructionReport(
        window_scores={},
        point_scores={},
        training=TrainingHistory((0.5,)),
        reproducibility=reproducibility,
        resolved_inputs=resolved_inputs,
    )
    destination = tmp_path / "run"

    with pytest.raises(ValueError, match=re.escape(path)):
        ReconstructionReportWriter(destination, provenance=provenance).write(result)

    assert not destination.exists()
