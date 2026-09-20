from __future__ import annotations

import importlib
import tempfile
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from detectiv.callbacks.base import BaseCallback
from detectiv.callbacks.mlflow import MlflowCallback
from detectiv.runs import RunArtifactWriter, TrainingEpochEvent
from detectiv.scenarios.results import ReconstructionScenarioResult


@dataclass(frozen=True)
class ReconstructionMlflowModelLogging:
    """Configure publication of a reconstruction model to MLflow."""

    name: str = "reconstruction_model"
    registered_model_name: str | None = None

    def __post_init__(self) -> None:
        if not self.name or self.registered_model_name == "":
            raise ValueError("MLflow model names must not be empty")


class ReconstructionMlflowCallback(BaseCallback[ReconstructionScenarioResult]):
    """Publish reconstruction-specific MLflow metrics and artifacts."""

    def __init__(
        self,
        tracking: MlflowCallback[ReconstructionScenarioResult],
        *,
        report_metrics_provider: Callable[[], Mapping[str, object]] | None = None,
        log_training_curve: bool = True,
        model_logging: ReconstructionMlflowModelLogging | None = None,
        model: torch.nn.Module | None = None,
        input_example: np.ndarray | None = None,
    ) -> None:
        if model_logging is not None and (model is None or input_example is None):
            raise ValueError("model logging requires model and input_example")
        self._tracking = tracking
        self._report_metrics_provider = report_metrics_provider
        self._log_training_curve = log_training_curve
        self._model_logging = model_logging
        self._model = model
        self._input_example = input_example

    @property
    def name(self) -> str:
        return "mlflow_reconstruction"

    def on_epoch_finished(self, event: TrainingEpochEvent) -> None:
        metrics = {"training.loss": event.training_loss}
        if event.validation_loss is not None:
            metrics["validation.loss"] = event.validation_loss
        metrics.update(
            {
                f"training.learning_rate.{index}": learning_rate
                for index, learning_rate in enumerate(event.learning_rates)
            }
        )
        metrics["training.epoch_seconds"] = event.elapsed_seconds
        self._tracking.log_metrics(metrics, step=event.epoch)

    def on_run_finished(self, result: ReconstructionScenarioResult) -> None:
        self._tracking.set_tags(
            {
                "training.epochs": str(len(result.training_losses)),
                "training.best_epoch": str(result.training.best_epoch),
                "training.best_validation_loss": str(
                    result.training.best_validation_loss
                ),
            }
        )
        if self._report_metrics_provider is not None:
            self._log_report_bundle(result)
        if self._log_training_curve:
            self._log_curve(result)
        if self._model_logging is not None:
            self._log_model()

    def _log_curve(self, result: ReconstructionScenarioResult) -> None:
        pyplot = importlib.import_module("matplotlib.pyplot")
        figure, axis = pyplot.subplots()
        axis.plot(
            range(len(result.training_losses)), result.training_losses, label="training"
        )
        if result.validation_losses:
            axis.plot(
                range(len(result.validation_losses)),
                result.validation_losses,
                label="validation",
            )
        axis.set(xlabel="Epoch", ylabel="Reconstruction loss")
        axis.legend()
        figure.tight_layout()
        self._tracking.log_figure(figure, "detectiv/training_loss.png")
        pyplot.close(figure)

    def _log_report_bundle(self, result: ReconstructionScenarioResult) -> None:
        if self._report_metrics_provider is None:
            return
        with tempfile.TemporaryDirectory() as directory:
            artifacts = RunArtifactWriter(
                Path(directory),
                provenance={
                    "configuration": self._tracking.configuration,
                    "dataset": self._tracking.dataset,
                    "parameters": self._tracking.parameters,
                },
            ).write(result, metrics=self._report_metrics_provider())
            self._tracking.log_artifacts(artifacts.manifest.parent)

    def _log_model(self) -> None:
        if (
            self._model_logging is None
            or self._model is None
            or self._input_example is None
        ):
            return
        values = np.asarray(self._input_example, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError(
                "model input_example must have shape (channels, height, width)"
            )
        batch = np.expand_dims(values, axis=0)
        model_to_log = deepcopy(self._model).to("cpu").eval()
        with torch.no_grad():
            output = model_to_log(torch.from_numpy(batch)).detach().numpy()
        signature = importlib.import_module("mlflow.models").infer_signature(
            batch, output
        )
        importlib.import_module("mlflow.pytorch").log_model(
            model_to_log,
            name=self._model_logging.name,
            input_example=batch,
            signature=signature,
            code_paths=[str(Path(__file__).parents[2])],
            registered_model_name=self._model_logging.registered_model_name,
            model_type="reconstruction",
            serialization_format="pickle",
        )
