import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from detectiv.scenarios import ReconstructionScenarioResult


@dataclass(frozen=True)
class RunArtifacts:
    manifest: Path
    point_scores: Path


class RunArtifactWriter:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def write(
        self,
        result: ReconstructionScenarioResult,
        *,
        metrics: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]]
        | None = None,
    ) -> RunArtifacts:
        self.directory.mkdir(parents=True, exist_ok=True)
        scores_path = self.directory / "point_scores.npz"
        manifest_path = self.directory / "run.json"
        arrays, score_keys = _score_arrays(result.point_scores)
        np.savez_compressed(scores_path, **arrays)  # type: ignore[arg-type]
        manifest: dict[str, object] = {
            "training_losses": result.training_losses,
            "validation_losses": result.validation_losses,
            "scores": {"path": scores_path.name, "keys": score_keys},
        }
        if metrics is not None:
            manifest["metrics"] = _plain_mapping(metrics)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return RunArtifacts(manifest_path, scores_path)


def _score_arrays(
    scores: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, dict[str, str]]]]:
    arrays: dict[str, np.ndarray] = {}
    keys: dict[str, dict[str, dict[str, str]]] = {}
    for index, (plan, propagations) in enumerate(scores.items()):
        keys[plan] = {}
        for propagation_index, (propagation, values) in enumerate(propagations.items()):
            keys[plan][propagation] = {}
            for series_index, (series_id, score) in enumerate(values.items()):
                key = f"score_{index}_{propagation_index}_{series_index}"
                arrays[key] = score
                keys[plan][propagation][series_id] = key
    return arrays, keys


def _plain_mapping(values: Mapping[str, object]) -> dict[str, object]:
    return {
        key: _plain_mapping(value) if isinstance(value, Mapping) else value
        for key, value in values.items()
    }
