import hashlib
import importlib
import json
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import torch

from detectiv.callbacks.base import BaseCallback
from detectiv.models.events import TrainingEpochEvent
from detectiv.scenarios.artifacts import RunArtifactWriter

if TYPE_CHECKING:
    from detectiv.scenarios.reconstruction import ReconstructionScenarioResult


@dataclass
class MlflowExperiment(AbstractContextManager["MlflowExperiment"]):
    experiment_name: str
    run_name: str | None = None
    parameters: Mapping[str, object] = field(default_factory=dict)
    configuration: Mapping[str, object] = field(default_factory=dict)
    dataset: Mapping[str, object] = field(default_factory=dict)
    tags: Mapping[str, str] = field(default_factory=dict)
    description: str | None = None
    metrics_provider: Callable[[], Mapping[str, object]] | None = None
    report_metrics_provider: Callable[[], Mapping[str, object]] | None = None
    model: "MlflowModelLogging | None" = None
    nested: bool = False
    tracking_uri: str | None = None
    log_system_metrics: bool = True
    log_training_curve: bool = True
    callback: "MlflowCallback" = field(init=False)
    _mlflow: Any = field(init=False, repr=False)
    _run_context: Any = field(init=False, repr=False)
    _active: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.experiment_name:
            raise ValueError("experiment_name must not be empty")
        self.callback = MlflowCallback(self)

    def __enter__(self) -> Self:
        self._mlflow = _load_mlflow()
        if self.tracking_uri is not None:
            self._mlflow.set_tracking_uri(self.tracking_uri)
        self._mlflow.set_experiment(self.experiment_name)
        self._run_context = self._mlflow.start_run(
            run_name=self.run_name,
            nested=self.nested,
            tags={
                **_lineage_tags(),
                **_dataset_tags(self.dataset),
                **self.tags,
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
                self._mlflow.log_dict(
                    dict(self.configuration), "detectiv/configuration.json"
                )
            if self.dataset:
                self._mlflow.log_dict(dict(self.dataset), "detectiv/dataset.json")
        except BaseException:
            self._active = False
            self._run_context.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, *args: object) -> bool | None:
        self._active = False
        self._run_context.__exit__(*args)
        return None

    def log_metrics(self, metrics: Mapping[str, object]) -> None:
        self._require_active()
        for name, value in _flat_values(metrics).items():
            if isinstance(value, bool | float | int):
                self._mlflow.log_metric(name, float(value))

    def set_metrics_provider(
        self, metrics_provider: Callable[[], Mapping[str, object]]
    ) -> None:
        if self._active:
            raise RuntimeError(
                "MLflow metrics provider must be set before the run starts"
            )
        self.metrics_provider = metrics_provider

    def set_report_metrics_provider(
        self, report_metrics_provider: Callable[[], Mapping[str, object]]
    ) -> None:
        if self._active:
            raise RuntimeError(
                "MLflow report metrics provider must be set before the run starts"
            )
        self.report_metrics_provider = report_metrics_provider

    def log_artifacts(
        self, directory: Path, *, artifact_path: str = "detectiv"
    ) -> None:
        self._require_active()
        if not directory.is_dir():
            raise ValueError(f"artifact directory does not exist: {directory}")
        self._mlflow.log_artifacts(str(directory), artifact_path=artifact_path)

    def log_model(self, model: torch.nn.Module, input_example: np.ndarray) -> None:
        if self.model is None:
            return
        self._require_active()
        values = np.asarray(input_example, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError(
                "model input_example must have shape (channels, height, width)"
            )
        batch = np.expand_dims(values, axis=0)
        model_to_log = deepcopy(model).to("cpu").eval()
        with torch.no_grad():
            output = model_to_log(torch.from_numpy(batch)).detach().numpy()
        signature = _load_mlflow_models().infer_signature(batch, output)
        _load_mlflow_pytorch().log_model(
            model_to_log,
            name=self.model.name,
            input_example=batch,
            signature=signature,
            code_paths=[str(_source_directory())],
            registered_model_name=self.model.registered_model_name,
            model_type="reconstruction",
            serialization_format="pickle",
        )

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError(
                "MLflow experiment is not active; use it as a context manager"
            )


class MlflowCallback(BaseCallback):
    def __init__(self, experiment: MlflowExperiment) -> None:
        self._experiment = experiment

    @property
    def name(self) -> str:
        return "mlflow"

    def on_run_started(self) -> None:
        self._experiment._require_active()

    def on_epoch_finished(self, event: TrainingEpochEvent) -> None:
        metrics = {
            "training.loss": event.training_loss,
        }
        if event.validation_loss is not None:
            metrics["validation.loss"] = event.validation_loss
        metrics.update(
            {
                f"training.learning_rate.{index}": learning_rate
                for index, learning_rate in enumerate(event.learning_rates)
            }
        )
        metrics["training.epoch_seconds"] = event.elapsed_seconds
        self._experiment._mlflow.log_metrics(metrics, step=event.epoch)

    def on_run_finished(self, result: "ReconstructionScenarioResult") -> None:
        self._experiment._mlflow.set_tags(
            {
                "training.epochs": str(len(result.training_losses)),
                "training.best_epoch": str(result.training.best_epoch),
                "training.best_validation_loss": str(
                    result.training.best_validation_loss
                ),
            }
        )
        if self._experiment.metrics_provider is not None:
            self._experiment.log_metrics(self._experiment.metrics_provider())
        if self._experiment.report_metrics_provider is not None:
            self._log_run_artifacts(result)
        if self._experiment.log_training_curve:
            self._log_training_curve(result)

    def on_run_failed(self, error: BaseException) -> None:
        if self._experiment._active:
            self._experiment._mlflow.set_tags(
                {
                    "detectiv.run_status": "failed",
                    "detectiv.error_type": type(error).__name__,
                }
            )

    def _log_training_curve(self, result: "ReconstructionScenarioResult") -> None:
        pyplot = _load_pyplot()
        figure, axis = pyplot.subplots()
        epochs = range(len(result.training_losses))
        axis.plot(epochs, result.training_losses, label="training")
        if result.validation_losses:
            axis.plot(
                range(len(result.validation_losses)),
                result.validation_losses,
                label="validation",
            )
        axis.set(xlabel="Epoch", ylabel="Reconstruction loss")
        axis.legend()
        figure.tight_layout()
        self._experiment._mlflow.log_figure(figure, "detectiv/training_loss.png")
        pyplot.close(figure)

    def _log_run_artifacts(self, result: "ReconstructionScenarioResult") -> None:
        if self._experiment.report_metrics_provider is None:
            return
        with tempfile.TemporaryDirectory() as directory:
            artifacts = RunArtifactWriter(
                Path(directory),
                provenance={
                    "configuration": self._experiment.configuration,
                    "dataset": self._experiment.dataset,
                    "parameters": self._experiment.parameters,
                },
            ).write(result, metrics=self._experiment.report_metrics_provider())
            self._experiment.log_artifacts(artifacts.manifest.parent)


@dataclass(frozen=True)
class MlflowModelLogging:
    name: str = "reconstruction_model"
    registered_model_name: str | None = None

    def __post_init__(self) -> None:
        if not self.name or self.registered_model_name == "":
            raise ValueError("MLflow model names must not be empty")


def _load_mlflow() -> Any:
    try:
        return importlib.import_module("mlflow")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "MLflow tracking requires `uv sync --extra experiment`."
        ) from error


def _load_mlflow_models() -> Any:
    return importlib.import_module("mlflow.models")


def _load_mlflow_pytorch() -> Any:
    return importlib.import_module("mlflow.pytorch")


def _load_pyplot() -> Any:
    return importlib.import_module("matplotlib.pyplot")


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
            ["git", *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode:
        return None
    return result.stdout.strip()


def _source_directory() -> Path:
    return Path(__file__).parents[2]
