import hashlib
import importlib
import json
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from detectiv.scenarios.reconstruction import ReconstructionScenarioResult


_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunArtifacts:
    manifest: Path
    point_scores: Path
    figures: tuple[Path, ...] = ()

    @classmethod
    def open(cls, directory: Path) -> "RunArtifacts":
        manifest = directory / "run.json"
        report = _read_report(manifest)
        scores = report.get("scores")
        if not isinstance(scores, Mapping) or not isinstance(scores.get("path"), str):
            raise ValueError("run report must define a point score artifact")
        artifacts = report.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("run report must define artifact checksums")
        paths = tuple(
            _artifact_path(directory, path)
            for path in artifacts
            if isinstance(path, str)
        )
        point_scores = _artifact_path(directory, scores["path"])
        figures = tuple(path for path in paths if path.parent.name == "figures")
        return cls(manifest, point_scores, figures)

    def read_report(self) -> Mapping[str, object]:
        return MappingProxyType(_read_report(self.manifest))

    def load_scores(self) -> Mapping[str, np.ndarray]:
        with np.load(self.point_scores, allow_pickle=False) as archive:
            scores = {
                name: _readonly(np.asarray(archive[name])) for name in archive.files
            }
        return MappingProxyType(scores)

    def verify(self) -> None:
        report = self.read_report()
        artifacts = report.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("run report must define artifact checksums")
        for relative_path, expected_hash in artifacts.items():
            if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
                raise ValueError("artifact checksums must map paths to SHA-256 strings")
            path = _artifact_path(self.manifest.parent, relative_path)
            if not path.is_file() or _sha256(path) != expected_hash:
                raise ValueError(
                    f"artifact checksum verification failed: {relative_path}"
                )


class RunArtifactWriter:
    def __init__(
        self,
        directory: Path,
        *,
        provenance: Mapping[str, object] | None = None,
        visualize: bool = False,
    ) -> None:
        self.directory = directory
        self.provenance = dict(provenance or {})
        self.visualize = visualize

    def write(
        self,
        result: "ReconstructionScenarioResult",
        *,
        metrics: Mapping[str, object] | None = None,
    ) -> RunArtifacts:
        self.directory.mkdir(parents=True, exist_ok=True)
        scores_path = self.directory / "point_scores.npz"
        manifest_path = self.directory / "run.json"
        arrays, score_keys = _score_arrays(result.point_scores)
        np.savez_compressed(scores_path, **arrays)  # type: ignore[arg-type]
        figures = _write_figures(self.directory, result) if self.visualize else ()
        manifest: dict[str, object] = {
            "schema_version": _REPORT_SCHEMA_VERSION,
            "runtime": _runtime_provenance(),
            "provenance": _plain_mapping(self.provenance),
            "training": {
                "losses": result.training_losses,
                "validation_losses": result.validation_losses,
                "best_epoch": result.training.best_epoch,
                "best_validation_loss": result.training.best_validation_loss,
            },
            "scores": {"path": scores_path.name, "keys": score_keys},
            "artifacts": _artifact_manifest(self.directory, (scores_path, *figures)),
        }
        if metrics is not None:
            manifest["metrics"] = _plain_mapping(metrics)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return RunArtifacts(manifest_path, scores_path, figures)


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


def _runtime_provenance() -> dict[str, str]:
    return {
        "detectiv_version": importlib.import_module("detectiv").__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def _artifact_manifest(directory: Path, paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(path.relative_to(directory)): _sha256(path) for path in paths}


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _read_report(path: Path) -> dict[str, object]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read run report: {path}") from error
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != _REPORT_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported run report: {path}")
    return report


def _artifact_path(directory: Path, relative_path: str) -> Path:
    root = directory.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"artifact path escapes the run directory: {relative_path}")
    return path


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


def _write_figures(
    directory: Path, result: "ReconstructionScenarioResult"
) -> tuple[Path, ...]:
    pyplot = _load_pyplot()
    figures_directory = directory / "figures"
    figures_directory.mkdir(exist_ok=True)
    paths = [_write_training_figure(pyplot, figures_directory, result)]
    for plan_index, propagations in enumerate(result.point_scores.values()):
        for propagation_index, values in enumerate(propagations.values()):
            for series_index, scores in enumerate(values.values()):
                paths.append(
                    _write_score_figure(
                        pyplot,
                        figures_directory,
                        scores,
                        f"score_{plan_index}_{propagation_index}_{series_index}.png",
                    )
                )
    return tuple(paths)


def _write_training_figure(
    pyplot: Any, directory: Path, result: "ReconstructionScenarioResult"
) -> Path:
    figure, axis = pyplot.subplots()
    axis.plot(result.training_losses, label="training")
    if result.validation_losses:
        axis.plot(result.validation_losses, label="validation")
    axis.set(xlabel="Epoch", ylabel="Reconstruction loss")
    axis.legend()
    figure.tight_layout()
    path = directory / "training_loss.png"
    figure.savefig(path)
    pyplot.close(figure)
    return path


def _write_score_figure(
    pyplot: Any, directory: Path, scores: np.ndarray, filename: str
) -> Path:
    figure, axis = pyplot.subplots()
    axis.plot(scores)
    axis.set(xlabel="Time index", ylabel="Anomaly score")
    figure.tight_layout()
    path = directory / filename
    figure.savefig(path)
    pyplot.close(figure)
    return path


def _load_pyplot() -> Any:
    try:
        return importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Run visualizations require `uv sync --extra experiment`."
        ) from error
