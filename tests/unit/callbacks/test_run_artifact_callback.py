from pathlib import Path

import numpy as np
import pytest

from detectiv.callbacks import RunArtifactCallback
from detectiv.models.autoencoders import TrainingHistory
from detectiv.runs import RunContext, RunIdentity
from detectiv.scenarios import ReconstructionScenarioResult


def test_artifact_callback_writes_the_completed_run(tmp_path: Path) -> None:
    callback = RunArtifactCallback(
        tmp_path,
        metrics_provider=lambda: {"test": {"score": 0.9}},
        provenance={"seed": 42},
    )
    result = ReconstructionScenarioResult(
        window_scores={},
        point_scores={"window": {"mean": np.array([1.0])}},
        training=TrainingHistory((0.5,)),
    )

    callback.on_run_started()
    callback.on_run_finished(result)

    assert callback.artifacts is not None
    assert callback.artifacts.manifest.is_file()
    assert callback.artifacts.point_scores.is_file()


def test_artifact_callback_uses_run_id_default_and_links_its_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    callback = RunArtifactCallback()
    context = RunContext(RunIdentity("detectiv-id"))
    context.register_mlflow("native-run", "mlruns:/native-run")
    result = ReconstructionScenarioResult(
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
    assert context.completed_run.artifact_location == callback.artifacts.manifest.parent
