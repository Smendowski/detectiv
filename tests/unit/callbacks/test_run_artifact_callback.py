from pathlib import Path

import numpy as np

from detectiv.callbacks import RunArtifactCallback
from detectiv.models.autoencoders import TrainingHistory
from detectiv.scenarios import ReconstructionScenarioResult


def test_artifact_callback_writes_the_completed_run(tmp_path: Path) -> None:
    callback = RunArtifactCallback(
        tmp_path,
        metrics_provider=lambda: {"test": {"score": 0.9}},
        provenance={"seed": 42},
    )
    result = ReconstructionScenarioResult(
        window_scores={},
        point_scores={"window": {"mean": {"series": np.array([1.0])}}},
        training=TrainingHistory((0.5,)),
    )

    callback.on_run_started()
    callback.on_run_finished(result)

    assert callback.artifacts is not None
    assert callback.artifacts.manifest.is_file()
    assert callback.artifacts.point_scores.is_file()
