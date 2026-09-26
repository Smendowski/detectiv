import hashlib
import importlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class MLflowTracker:
    """Own one reusable MLflow run configuration and its active run."""

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
        nested: bool = False,
        log_system_metrics: bool = True,
    ) -> None:
        """Configure MLflow metadata shared by every tracked run."""
        if not experiment_name:
            raise ValueError("experiment_name must not be empty")
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.parameters = dict(parameters or {})
        self.configuration = dict(configuration or {})
        self.dataset = dict(dataset or {})
        self.tags = dict(tags or {})
        self.description = description
        self.nested = nested
        self.log_system_metrics = log_system_metrics
        self._mlflow: Any = None
        self._run: Any = None
        self._active = False

    @property
    def active(self) -> bool:
        """Return whether this tracker currently owns an open MLflow run."""
        return self._active

    def start(self) -> tuple[str, str | None]:
        """Start a configured MLflow run and return its ID and artifact URI."""
        if self._active:
            raise RuntimeError("MLflow tracker already has an active run")

        self._mlflow = _load_mlflow()
        self._mlflow.set_experiment(self.experiment_name)

        tags = dict(self.tags)
        tags.update(_lineage_tags())
        tags.update(_dataset_tags(self.dataset))
        if self.description is not None:
            tags["mlflow.note.content"] = self.description

        self._run = self._mlflow.start_run(
            run_name=self.run_name,
            nested=self.nested,
            tags=tags,
            log_system_metrics=self.log_system_metrics,
        )
        self._active = True

        try:
            parameters = {
                name: _parameter_value(value)
                for name, value in _flatten(self.parameters).items()
            }
            self._mlflow.log_params(parameters)
            if self.configuration:
                self._mlflow.log_dict(self.configuration, "detectiv/configuration.json")
            if self.dataset:
                self._mlflow.log_dict(self.dataset, "detectiv/dataset.json")
            info = self._run.info
            return str(info.run_id), getattr(info, "artifact_uri", None)
        except BaseException as error:
            self.close(error)
            raise

    def close(self, error: BaseException | None = None) -> None:
        """Close the active run with an optional failure."""
        if not self._active:
            return
        status = "FINISHED" if error is None else "FAILED"
        try:
            self._mlflow.end_run(status=status)
        finally:
            self._active = False
            self._run = None

    def log_metrics(
        self, metrics: Mapping[str, object], *, step: int | None = None
    ) -> None:
        """Log flattened numeric metric leaves in the active run."""
        self._require_active()
        for name, value in _flatten(metrics).items():
            if isinstance(value, bool | float | int):
                self._mlflow.log_metric(name, float(value), step=step)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Set tags in the active run."""
        self._require_active()
        self._mlflow.set_tags(tags)

    def log_artifacts(
        self, directory: Path, *, artifact_path: str = "detectiv"
    ) -> None:
        """Upload an existing directory to the active run."""
        self._require_active()
        if not directory.is_dir():
            raise ValueError(f"artifact directory does not exist: {directory}")
        self._mlflow.log_artifacts(str(directory), artifact_path=artifact_path)

    def log_pytorch_model(
        self,
        model: object,
        *,
        name: str,
        input_example: object,
        output_example: object,
        registered_model_name: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Publish a PyTorch model with an inferred MLflow signature."""
        self._require_active()
        signature = importlib.import_module("mlflow.models").infer_signature(
            input_example, output_example
        )
        importlib.import_module("mlflow.pytorch").log_model(
            model,
            name=name,
            input_example=input_example,
            signature=signature,
            code_paths=[str(Path(__file__).parent.parent)],
            registered_model_name=registered_model_name,
            metadata=dict(metadata or {}),
            serialization_format="pickle",
        )

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("MLflow tracker is not active")


def _load_mlflow() -> Any:
    try:
        return importlib.import_module("mlflow")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "MLflow tracking requires `uv sync --extra experiment`."
        ) from error


def _flatten(values: Mapping[str, object], prefix: str = "") -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in values.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, Mapping):
            result.update(_flatten(value, key))
        elif value is not None:
            result[key] = value
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
