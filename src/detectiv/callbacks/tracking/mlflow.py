from __future__ import annotations

import hashlib
import importlib
import json
import platform
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypedDict

from detectiv.callbacks.base import BaseCallback
from detectiv.runs import RunContext


class MlflowOptions(TypedDict, total=False):
    """Optional generic configuration shared by MLflow tracking callbacks."""

    run_name: str | None
    parameters: Mapping[str, object] | None
    configuration: Mapping[str, object] | None
    dataset: Mapping[str, object] | None
    tags: Mapping[str, str] | None
    description: str | None
    tracking_metrics_provider: Callable[[], Mapping[str, object]] | None
    artifact_directories: Mapping[Path, str] | None
    nested: bool
    tracking_uri: str | None
    log_system_metrics: bool


class MlflowCallback[T](BaseCallback[T]):
    """Track generic scenario metadata and lifecycle in an MLflow run.

    The callback owns the optional MLflow resource lifecycle. Register it in a
    scenario's ``callbacks`` tuple; no external context manager is needed.
    """

    def __init__(
        self,
        experiment_name: str = "experiments",
        *,
        run_name: str | None = None,
        parameters: Mapping[str, object] | None = None,
        configuration: Mapping[str, object] | None = None,
        dataset: Mapping[str, object] | None = None,
        tags: Mapping[str, str] | None = None,
        description: str | None = None,
        tracking_metrics_provider: Callable[[], Mapping[str, object]] | None = None,
        artifact_directories: Mapping[Path, str] | None = None,
        nested: bool = False,
        tracking_uri: str | None = None,
        log_system_metrics: bool = True,
    ) -> None:
        if not experiment_name:
            raise ValueError("experiment_name must not be empty")
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.parameters = dict(parameters or {})
        self.configuration = dict(configuration or {})
        self.dataset = dict(dataset or {})
        self.tags = dict(tags or {})
        self.description = description
        self.tracking_metrics_provider = tracking_metrics_provider
        self.artifact_directories = dict(artifact_directories or {})
        self.nested = nested
        self.tracking_uri = tracking_uri
        self.log_system_metrics = log_system_metrics
        self._mlflow: Any = None
        self._run_context: Any = None
        self._active = False
        self._failure: BaseException | None = None

    @property
    def name(self) -> str:
        """Return the fixed registration name `mlflow`.

        Returns:
            The callback registration name.
        """
        return "mlflow"

    def on_run_context(self, context: RunContext) -> None:
        """Retain the shared run context for MLflow lineage.

        Args:
            context: Shared run identity and output-registration context.
        """
        self._context = context

    def on_run_started(self) -> None:
        """Open and initialize the configured MLflow run.

        Raises:
            RuntimeError: If MLflow does not expose the newly started run.
        """
        self._mlflow = _load_mlflow()
        if self.tracking_uri is not None:
            self._mlflow.set_tracking_uri(self.tracking_uri)
        self._mlflow.set_experiment(self.experiment_name)
        self._run_context = self._mlflow.start_run(
            run_name=self.run_name,
            nested=self.nested,
            tags={
                **self.tags,
                **_lineage_tags(),
                **_dataset_tags(self.dataset),
                **(
                    {"mlflow.note.content": self.description}
                    if self.description is not None
                    else {}
                ),
            },
            log_system_metrics=self.log_system_metrics,
        )
        self._run_context.__enter__()
        self._active = True
        try:
            self._mlflow.log_params(_flat_values(self.parameters))
            if self.configuration:
                self._mlflow.log_dict(self.configuration, "detectiv/configuration.json")
            if self.dataset:
                self._mlflow.log_dict(self.dataset, "detectiv/dataset.json")
            active_run = self._mlflow.active_run()
            if active_run is None:
                raise RuntimeError("MLflow callback has no active run")
            info = active_run.info
            tags = {
                "detectiv.run_id": self._context.identity.run_id,
                "detectiv.run_status": "started",
            }
            if self._context.scenario_type is not None:
                tags["detectiv.scenario_type"] = self._context.scenario_type
            self._mlflow.set_tags(tags)
            self._context.register_mlflow(
                info.run_id, getattr(info, "artifact_uri", None)
            )
        except BaseException:
            self._active = False
            self._run_context.__exit__(*sys.exc_info())
            raise

    def on_run_finished(self, result: T) -> None:
        """Log configured success outputs and status.

        Args:
            result: Successful scenario result; generic tracking does not inspect it.
        """
        self._require_active()
        if self.tracking_metrics_provider is not None:
            self.log_metrics(self.tracking_metrics_provider())
        for directory, artifact_path in self.artifact_directories.items():
            self.log_artifacts(directory, artifact_path=artifact_path)
        self.set_tags({"detectiv.run_status": "succeeded"})
        return None

    def on_run_failed(self, error: BaseException) -> None:
        """Record a failed status for an active MLflow run.

        Args:
            error: Original scenario exception.
        """
        if self._active:
            self._failure = error
            self.set_tags(
                {
                    "detectiv.run_status": "failed",
                    "detectiv.error_type": type(error).__name__,
                }
            )

    def on_run_closed(self) -> None:
        """Close the active MLflow run with its recorded outcome."""
        if self._active:
            self._active = False
            if self._failure is None:
                self._run_context.__exit__(None, None, None)
            else:
                self._run_context.__exit__(
                    type(self._failure), self._failure, self._failure.__traceback__
                )

    def log_metrics(
        self, metrics: Mapping[str, object], *, step: int | None = None
    ) -> None:
        """Log flattened numeric metric leaves in the active MLflow run.

        Args:
            metrics: Nested metric values; numeric leaves are logged.
            step: Optional metric step.
        """
        self._require_active()
        for name, value in _flat_values(metrics).items():
            if isinstance(value, bool | float | int):
                self._mlflow.log_metric(name, float(value), step=step)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Set tags in the active MLflow run.

        Args:
            tags: Tag names and values.
        """
        self._require_active()
        self._mlflow.set_tags(tags)

    def log_artifacts(
        self, directory: Path, *, artifact_path: str = "detectiv"
    ) -> None:
        """Upload an existing directory under an MLflow artifact path.

        Args:
            directory: Existing local directory to upload.
            artifact_path: Destination path within the MLflow run.

        Raises:
            ValueError: If `directory` does not exist or is not a directory.
        """
        self._require_active()
        if not directory.is_dir():
            raise ValueError(f"artifact directory does not exist: {directory}")
        self._mlflow.log_artifacts(str(directory), artifact_path=artifact_path)

    def log_figure(self, figure: object, artifact_file: str) -> None:
        """Upload an already-created figure to the active MLflow run.

        Args:
            figure: Figure supported by MLflow's figure logger.
            artifact_file: Destination file within the MLflow run.
        """
        self._require_active()
        self._mlflow.log_figure(figure, artifact_file)

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("MLflow callback is not active")


def _load_mlflow() -> Any:
    try:
        return importlib.import_module("mlflow")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "MLflow tracking requires `uv sync --extra experiment`."
        ) from error


def _flat_values(values: Mapping[str, object], prefix: str = "") -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in values.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, Mapping):
            result.update(_flat_values(value, key))
        elif value is not None:
            result[key] = _parameter_value(value)
    return result


def _parameter_value(value: object) -> bool | float | int | str:
    if isinstance(value, bool | float | int | str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _dataset_tags(dataset: Mapping[str, object]) -> dict[str, str]:
    if not dataset:
        return {}
    encoded = json.dumps(dataset, sort_keys=True, default=str, separators=(",", ":"))
    return {
        "detectiv.dataset.manifest_sha256": hashlib.sha256(encoded.encode()).hexdigest()
    }


def _lineage_tags() -> dict[str, str]:
    tags = {
        "detectiv.python": platform.python_version(),
        "detectiv.platform": platform.platform(),
        "detectiv.implementation": platform.python_implementation(),
        "detectiv.executable": sys.executable,
    }
    if commit := _git("rev-parse", "HEAD"):
        tags["detectiv.git.commit"] = commit
        tags["detectiv.git.dirty"] = str(bool(_git("status", "--porcelain")))
    return tags


def _git(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments], check=False, capture_output=True, text=True
        )
    except OSError:
        return None
    if result.returncode:
        return None
    return result.stdout.strip()
