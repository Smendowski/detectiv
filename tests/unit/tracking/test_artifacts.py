import json
from pathlib import Path

import numpy as np

from detectiv.data import WindowReference
from detectiv.models.autoencoders import TrainingHistory
from detectiv.scenarios import ReconstructionScenarioResult
from detectiv.scoring import WindowScoreBatch
from detectiv.tracking import RunArtifactWriter


def test_run_artifact_writer_separates_manifest_and_point_scores(
    tmp_path: Path,
) -> None:
    result = ReconstructionScenarioResult(
        window_scores={
            "window": WindowScoreBatch(
                np.array([1.0]), (WindowReference("series", 0, 3, 3),)
            )
        },
        point_scores={"window": {"mean": {"series": np.array([1.0, 2.0, 3.0])}}},
        training=TrainingHistory((0.5,)),
    )

    artifacts = RunArtifactWriter(tmp_path).write(
        result,
        metrics={"window": {"mean": {"series": {"AUC-PR": 0.8}}}},
    )

    manifest = json.loads(artifacts.manifest.read_text())
    scores = np.load(artifacts.point_scores)
    assert manifest["training_losses"] == [0.5]
    assert manifest["validation_losses"] == []
    assert manifest["scores"]["keys"] == {"window": {"mean": {"series": "score_0_0_0"}}}
    assert manifest["metrics"] == {"window": {"mean": {"series": {"AUC-PR": 0.8}}}}
    np.testing.assert_allclose(scores["score_0_0_0"], [1.0, 2.0, 3.0])
