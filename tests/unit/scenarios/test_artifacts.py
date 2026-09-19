import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from detectiv.models.autoencoders import TrainingHistory
from detectiv.scenarios import (
    ReconstructionScenarioResult,
    RunArtifacts,
    RunArtifactWriter,
)
from detectiv.scoring import WindowScoreBatch
from detectiv.time_series.windowing import WindowReference


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

    artifacts = RunArtifactWriter(
        tmp_path, provenance={"dataset": {"name": "synthetic"}}
    ).write(
        result,
        metrics={"window": {"mean": {"series": {"AUC-PR": 0.8}}}},
    )

    manifest = json.loads(artifacts.manifest.read_text())
    scores = np.load(artifacts.point_scores)
    assert manifest["schema_version"] == 1
    assert manifest["training"] == {
        "losses": [0.5],
        "validation_losses": [],
        "best_epoch": None,
        "best_validation_loss": None,
    }
    assert manifest["provenance"] == {"dataset": {"name": "synthetic"}}
    assert manifest["scores"]["keys"] == {"window": {"mean": {"series": "score_0_0_0"}}}
    assert manifest["metrics"] == {"window": {"mean": {"series": {"AUC-PR": 0.8}}}}
    assert (
        manifest["artifacts"]["point_scores.npz"]
        == hashlib.file_digest(artifacts.point_scores.open("rb"), "sha256").hexdigest()
    )
    np.testing.assert_allclose(scores["score_0_0_0"], [1.0, 2.0, 3.0])


def test_run_artifact_writer_creates_opt_in_figures(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib.pyplot")
    result = ReconstructionScenarioResult(
        window_scores={},
        point_scores={"window": {"mean": {"series": np.array([1.0, 2.0])}}},
        training=TrainingHistory((0.5,), validation_losses=(0.4,)),
    )

    artifacts = RunArtifactWriter(tmp_path, visualize=True).write(result)

    assert [path.relative_to(tmp_path).as_posix() for path in artifacts.figures] == [
        "figures/training_loss.png",
        "figures/score_0_0_0.png",
    ]


def test_run_artifacts_open_verify_and_load_scores(tmp_path: Path) -> None:
    result = ReconstructionScenarioResult(
        window_scores={},
        point_scores={"window": {"mean": {"series": np.array([1.0, 2.0])}}},
        training=TrainingHistory((0.5,)),
    )
    RunArtifactWriter(tmp_path).write(result)

    artifacts = RunArtifacts.open(tmp_path)

    artifacts.verify()
    scores = artifacts.load_scores()
    np.testing.assert_allclose(scores["score_0_0_0"], [1.0, 2.0])
    assert not scores["score_0_0_0"].flags.writeable


def test_run_artifacts_detect_corrupted_scores(tmp_path: Path) -> None:
    result = ReconstructionScenarioResult(
        window_scores={},
        point_scores={"window": {"mean": {"series": np.array([1.0])}}},
        training=TrainingHistory((0.5,)),
    )
    artifacts = RunArtifactWriter(tmp_path).write(result)
    artifacts.point_scores.write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="checksum"):
        RunArtifacts.open(tmp_path).verify()
