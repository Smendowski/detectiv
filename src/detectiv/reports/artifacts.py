from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from detectiv.reports.reconstruction import ReconstructionReport
from detectiv.runs.identity import CompletedRunSummary, RunIdentity, RunOutputLocator
from detectiv.runs.metadata import JSONValue

_REPORT_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class ReportArtifacts:
    """Paths and verified accessors for one persisted report bundle."""

    manifest: Path
    point_scores: Path
    figures: tuple[Path, ...] = ()

    @classmethod
    def open(cls, directory: Path) -> ReportArtifacts:
        """Open a schema-compatible report bundle without hashing its files.

        Args:
            directory: Directory containing ``run.json`` and its artifacts.

        Returns:
            A handle to the manifest, point scores, and figures.

        Raises:
            ValueError: If the manifest is invalid or omits the score checksum.
        """
        manifest = directory / "run.json"
        report = _read_report(manifest)
        scores = report.get("scores")
        if not isinstance(scores, Mapping) or not isinstance(scores.get("path"), str):
            raise ValueError("run report must define a point score artifact")
        artifacts = report.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("run report must define artifact checksums")
        _validate_score_checksum(scores["path"], artifacts)
        paths = tuple(
            _artifact_path(directory, path)
            for path in artifacts
            if isinstance(path, str)
        )
        point_scores = _artifact_path(directory, scores["path"])
        figures = tuple(path for path in paths if path.parent.name == "figures")
        return cls(manifest, point_scores, figures)

    @classmethod
    def open_verified(cls, directory: Path) -> ReportArtifacts:
        """Open a report bundle and verify every declared artifact.

        Args:
            directory: Directory containing ``run.json`` and its artifacts.

        Returns:
            A verified handle to the report bundle.

        Raises:
            ValueError: If the manifest or an artifact fails validation.
        """
        artifacts = cls.open(directory)
        artifacts.verify()
        return artifacts

    def read_report(self) -> Mapping[str, object]:
        """Read the persisted JSON report without verifying artifact contents.

        Returns:
            A read-only mapping containing the persisted report record.

        Raises:
            ValueError: If the manifest cannot be read or has an unsupported schema.
        """
        return MappingProxyType(_read_report(self.manifest))

    def read_verified_report(self) -> Mapping[str, object]:
        """Verify the bundle and read its persisted JSON report.

        Returns:
            A read-only mapping containing the persisted report record.

        Raises:
            ValueError: If the manifest or an artifact fails validation.
        """
        self.verify()
        return self.read_report()

    def read_completed_run(self) -> CompletedRunSummary:
        """Read linked Detectiv and optional MLflow identifiers from the manifest.

        Returns:
            The run identity and known local and MLflow output locations.

        Raises:
            ValueError: If the manifest has no valid Detectiv run identity.
        """
        report = self.read_report()
        run = report.get("run")
        if not isinstance(run, Mapping) or not isinstance(run.get("detectiv_id"), str):
            raise ValueError("run report does not define Detectiv run identity")
        mlflow_run_id = run.get("mlflow_run_id")
        if mlflow_run_id is not None and not isinstance(mlflow_run_id, str):
            raise ValueError("run report defines an invalid MLflow run ID")
        return CompletedRunSummary(
            RunIdentity(run["detectiv_id"]),
            local_artifacts=RunOutputLocator(location=self.manifest.parent),
            mlflow=(
                None
                if mlflow_run_id is None
                else RunOutputLocator(native_run_id=mlflow_run_id)
            ),
        )

    def load_scores(self) -> Mapping[str, np.ndarray]:
        """Load the stored score arrays by their stable archive keys.

        Returns:
            Read-only score arrays keyed by names such as ``score_0_0``.
        """
        with np.load(self.point_scores, allow_pickle=False) as archive:
            scores = {
                name: _readonly(np.asarray(archive[name])) for name in archive.files
            }
        return MappingProxyType(scores)

    def load_verified_scores(self) -> Mapping[str, np.ndarray]:
        """Verify the bundle before loading its score arrays.

        Returns:
            Read-only score arrays keyed by their stable archive keys.

        Raises:
            ValueError: If the manifest or an artifact fails validation.
        """
        self.verify()
        return self.load_scores()

    def verify(self) -> None:
        """Verify the checksum of every artifact declared by the manifest.

        Raises:
            ValueError: If checksums are invalid, missing, or do not match files.
        """
        report = self.read_report()
        artifacts = report.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("run report must define artifact checksums")
        scores = report.get("scores")
        if not isinstance(scores, Mapping) or not isinstance(scores.get("path"), str):
            raise ValueError("run report must define a point score artifact")
        _validate_score_checksum(scores["path"], artifacts)
        for relative_path, expected_hash in artifacts.items():
            if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
                raise ValueError("artifact checksums must map paths to SHA-256 strings")
            path = _artifact_path(self.manifest.parent, relative_path)
            if not path.is_file() or _sha256(path) != expected_hash:
                raise ValueError(
                    f"artifact checksum verification failed: {relative_path}"
                )


class ReconstructionReportWriter:
    """Atomically persist a reconstruction report and its point scores.

    Args:
        directory: Destination directory for the complete report bundle.
        provenance: Additional JSON-compatible bundle provenance.
        visualize: Whether to render training and point-score figures.
        overwrite: Whether to replace an existing non-empty destination.
    """

    def __init__(
        self,
        directory: Path,
        *,
        provenance: Mapping[str, object] | None = None,
        visualize: bool = False,
        overwrite: bool = False,
    ) -> None:
        self.directory = directory
        self.provenance = dict(provenance or {})
        self.visualize = visualize
        self.overwrite = overwrite

    def write(
        self,
        report: ReconstructionReport,
        *,
        completed_run: CompletedRunSummary | None = None,
    ) -> ReportArtifacts:
        """Write one reconstruction report as a schema-versioned bundle.

        Args:
            report: Completed reconstruction report to persist.
            completed_run: Optional execution identity linked in the manifest.

        Returns:
            Paths and accessors for the atomically published bundle.

        Raises:
            FileExistsError: If the destination is non-empty without overwrite.
            ValueError: If report or provenance values cannot be persisted safely.
        """
        _validate_json(self.provenance, "provenance")
        _validate_json(report.reproducibility, "reproducibility")
        _validate_json(report.resolved_inputs, "resolved_inputs")
        _validate_json(report.metrics, "metrics")
        self._prepare_destination()
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{self.directory.name}.", dir=self.directory.parent
            )
        )

        try:
            artifacts = self._write(staging, report, completed_run=completed_run)
            self._publish(staging)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        return ReportArtifacts(
            self.directory / artifacts.manifest.name,
            self.directory / artifacts.point_scores.name,
            tuple(
                self.directory / path.relative_to(staging) for path in artifacts.figures
            ),
        )

    def _write(
        self,
        directory: Path,
        report: ReconstructionReport,
        *,
        completed_run: CompletedRunSummary | None,
    ) -> ReportArtifacts:
        scores_path = directory / "point_scores.npz"
        manifest_path = directory / "run.json"
        arrays, score_keys = _score_arrays(report.point_scores)
        np.savez_compressed(scores_path, **arrays)  # type: ignore[arg-type]
        figures = _write_figures(directory, report) if self.visualize else ()
        manifest: dict[str, object] = {
            "schema_version": _REPORT_SCHEMA_VERSION,
            "runtime": _runtime_provenance(),
            "provenance": _json_mapping(self.provenance),
            **_json_mapping(report.record()),
            "scores": {"path": scores_path.name, "keys": score_keys},
            "artifacts": _artifact_manifest(directory, (scores_path, *figures)),
        }
        if completed_run is not None:
            manifest["run"] = {
                "detectiv_id": completed_run.run_id,
                "mlflow_run_id": completed_run.mlflow_run_id,
            }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return ReportArtifacts(manifest_path, scores_path, figures)

    def _prepare_destination(self) -> None:
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        if not self.directory.exists():
            return
        if not self.directory.is_dir():
            raise ValueError(
                f"artifact destination is not a directory: {self.directory}"
            )
        if any(self.directory.iterdir()) and not self.overwrite:
            raise FileExistsError(
                "artifact destination is not empty: "
                f"{self.directory}; use overwrite=True"
            )

    def _publish(self, staging: Path) -> None:
        if self.directory.exists():
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{self.directory.name}.previous.",
                    dir=self.directory.parent,
                )
            )
            backup.rmdir()
            os.replace(self.directory, backup)
            try:
                os.replace(staging, self.directory)
            except BaseException:
                os.replace(backup, self.directory)
                raise
            shutil.rmtree(backup)
            return
        os.replace(staging, self.directory)


def _score_arrays(
    scores: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, str]]]:
    arrays: dict[str, np.ndarray] = {}
    keys: dict[str, dict[str, str]] = {}
    for index, (plan, propagations) in enumerate(scores.items()):
        keys[plan] = {}
        for propagation_index, (propagation, score) in enumerate(propagations.items()):
            if (
                not np.issubdtype(score.dtype, np.number)
                or not np.isfinite(score).all()
            ):
                raise ValueError(
                    "point scores must contain only finite numeric values: "
                    f"{plan}.{propagation}"
                )
            key = f"score_{index}_{propagation_index}"
            arrays[key] = score
            keys[plan][propagation] = key
    return arrays, keys


def _validate_json(value: object, path: str) -> None:
    if isinstance(value, float | np.floating) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    if value is None or isinstance(value, bool | int | float | str):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} has a non-string key: {key!r}")
            _validate_json(item, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
        return
    raise ValueError(f"{path} is not JSON-compatible: {type(value).__name__}")


def _json_mapping(values: Mapping[str, object]) -> dict[str, JSONValue]:
    return {key: _json_value(value) for key, value in values.items()}


def _json_value(value: object) -> JSONValue:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_value(item) for item in value]
    raise ValueError(f"JSON value is not compatible: {type(value).__name__}")


def _validate_score_checksum(
    score_path: str, artifacts: Mapping[object, object]
) -> None:
    checksum = artifacts.get(score_path)
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("run report must checksum the point score artifact")
    try:
        int(checksum, 16)
    except ValueError as error:
        raise ValueError("run report must checksum the point score artifact") from error


def _runtime_provenance() -> dict[str, str]:
    root = Path(__file__).parents[3]
    lockfile = root / "uv.lock"
    return {
        "detectiv_version": importlib.import_module("detectiv").__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "source_revision": _git("rev-parse", "HEAD") or "unknown",
        "source_dirty": str(bool(_git("status", "--porcelain"))),
        "dependency_lock_sha256": (
            _sha256(lockfile) if lockfile.is_file() else "unknown"
        ),
    }


def _artifact_manifest(directory: Path, paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(path.relative_to(directory)): _sha256(path) for path in paths}


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _git(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


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


def _write_figures(directory: Path, report: ReconstructionReport) -> tuple[Path, ...]:
    pyplot = _load_pyplot()
    figures_directory = directory / "figures"
    figures_directory.mkdir(exist_ok=True)
    paths = [_write_training_figure(pyplot, figures_directory, report)]
    for plan_index, propagations in enumerate(report.point_scores.values()):
        for propagation_index, scores in enumerate(propagations.values()):
            paths.append(
                _write_score_figure(
                    pyplot,
                    figures_directory,
                    scores,
                    f"score_{plan_index}_{propagation_index}.png",
                )
            )
    return tuple(paths)


def _write_training_figure(
    pyplot: Any, directory: Path, report: ReconstructionReport
) -> Path:
    figure, axis = pyplot.subplots()
    axis.plot(report.training_losses, label="training")
    if report.validation_losses:
        axis.plot(report.validation_losses, label="validation")
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
