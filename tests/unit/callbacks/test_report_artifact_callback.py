from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from detectiv.callbacks import MetricsCallback, ReportArtifactCallback
from detectiv.models.autoencoders import TrainingHistory
from detectiv.reports import ReconstructionReport, RunContext
from detectiv.scenarios import BaseScenario


class _Evaluator:
    def evaluate(
        self, point_scores: np.ndarray, labels: np.ndarray
    ) -> Mapping[str, float]:
        return {"mean": float(point_scores.mean())}


class _Scenario(BaseScenario[ReconstructionReport]):
    def __init__(
        self, callbacks: tuple[MetricsCallback | ReportArtifactCallback, ...]
    ) -> None:
        super().__init__(callbacks=callbacks)

    def _run(self) -> ReconstructionReport:
        return ReconstructionReport(
            window_scores={},
            point_scores={"window": {"mean": np.array([0.1, 0.9])}},
            point_labels=np.array([0, 1]),
            training=TrainingHistory((0.5,)),
        )


def test_artifact_callback_writes_the_completed_run(tmp_path: Path) -> None:
    callback = ReportArtifactCallback(
        str(tmp_path),
        provenance={"seed": 42},
    )
    _Scenario((callback,)).run()

    assert callback.artifacts is not None
    assert callback.artifacts.manifest.is_file()
    assert callback.artifacts.point_scores.is_file()


def test_artifact_callback_uses_run_id_default_and_links_its_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    callback = ReportArtifactCallback()
    context = RunContext("detectiv-id")
    context.register_mlflow("native-run", "mlruns:/native-run")
    result = ReconstructionReport(
        window_scores={},
        point_scores={"window": {"mean": np.array([1.0])}},
        training=TrainingHistory((0.5,)),
    )

    callback.on_run_context(context)
    callback.on_run_started()
    callback.on_run_finished(result)

    assert callback.artifacts is not None
    assert callback.artifacts.manifest == Path("artifacts/detectiv-id/run.json")
    manifest_run = callback.artifacts.read_completed_run()
    assert manifest_run.run_id == "detectiv-id"
    assert manifest_run.mlflow_run_id == "native-run"
    assert manifest_run.mlflow_location == "mlruns:/native-run"
    assert context.completed_run.artifact_location == callback.artifacts.manifest.parent


def test_metrics_registered_before_artifacts_are_persisted(tmp_path: Path) -> None:
    publisher = ReportArtifactCallback(tmp_path)

    report = _Scenario((MetricsCallback(_Evaluator()), publisher)).run()

    assert report.metrics == {"window.mean.mean": 0.5}
    assert publisher.artifacts is not None
    assert publisher.artifacts.read_report()["metrics"] == report.metrics
