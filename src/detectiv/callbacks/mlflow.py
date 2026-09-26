from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from detectiv.callbacks.base import BaseCallback
from detectiv.models.autoencoders import TrainingEpochEvent
from detectiv.reports import ReconstructionReport, RunContext
from detectiv.tracking import MLflowTracker


class BaseMLflowCallback[T](BaseCallback[T]):
    """Adapt Detectiv's callback lifecycle to an MLflow tracker."""

    def __init__(
        self,
        tracker: MLflowTracker,
        *,
        metrics_provider: Callable[[T], Mapping[str, object]] | None = None,
        artifact_directories: Mapping[Path, str] | None = None,
    ) -> None:
        """Configure final metrics and existing artifact directories."""
        self.tracker = tracker
        self.metrics_provider = metrics_provider
        self.artifact_directories = dict(artifact_directories or {})
        self._context: RunContext | None = None
        self._failure: BaseException | None = None

    @property
    def name(self) -> str:
        """Return the fixed registration name `mlflow`."""
        return "mlflow"

    def on_run_context(self, context: RunContext) -> None:
        """Retain the Detectiv run context for MLflow lineage."""
        self._context = context

    def on_run_started(self) -> None:
        """Start tracking and register the native MLflow run."""
        if self._context is None:
            raise RuntimeError("MLflow callback did not receive a run context")

        self._failure = None
        try:
            run_id, artifact_uri = self.tracker.start()
            tags = {
                "detectiv.run_id": self._context.run_id,
                "detectiv.run_status": "started",
            }
            if self._context.scenario_type is not None:
                tags["detectiv.scenario_type"] = self._context.scenario_type
            self.tracker.set_tags(tags)
            self._context.register_mlflow(run_id, artifact_uri)
        except BaseException as error:
            self.tracker.close(error)
            raise

    def on_run_finished(self, result: T) -> None:
        """Publish configured final metrics and artifacts."""
        if self.metrics_provider is not None:
            self.tracker.log_metrics(self.metrics_provider(result))
        for directory, artifact_path in self.artifact_directories.items():
            self.tracker.log_artifacts(directory, artifact_path=artifact_path)
        self.tracker.set_tags({"detectiv.run_status": "succeeded"})

    def on_run_failed(self, error: BaseException) -> None:
        """Record a failed status for an active MLflow run."""
        if self.tracker.active:
            self._failure = error
            self.tracker.set_tags(
                {
                    "detectiv.run_status": "failed",
                    "detectiv.error_type": type(error).__name__,
                }
            )

    def on_run_closed(self) -> None:
        """Close the MLflow run with its recorded outcome."""
        try:
            self.tracker.close(self._failure)
        finally:
            self._failure = None
            self._context = None


class MLflowCallback(BaseMLflowCallback[ReconstructionReport]):
    """Track reconstruction training and results in MLflow."""

    def __init__(
        self,
        tracker: MLflowTracker,
        *,
        artifact_directories: Mapping[Path, str] | None = None,
        model: torch.nn.Module | None = None,
        input_example: np.ndarray | None = None,
        model_name: str = "reconstruction_model",
        registered_model_name: str | None = None,
    ) -> None:
        """Configure reconstruction artifacts and optional model publication."""
        if (model is None) != (input_example is None):
            raise ValueError("model logging requires model and input_example")
        if not model_name or registered_model_name == "":
            raise ValueError("MLflow model names must not be empty")
        super().__init__(tracker, artifact_directories=artifact_directories)
        self.model = model
        self.input_example = input_example
        self.model_name = model_name
        self.registered_model_name = registered_model_name

    def on_epoch_finished(self, event: TrainingEpochEvent) -> None:
        """Log losses, learning rates, and duration for one epoch."""
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
        self.tracker.log_metrics(metrics, step=event.epoch)

    def on_run_finished(self, result: ReconstructionReport) -> None:
        """Log reconstruction training, metrics, artifacts, and optional model."""
        tags = {"training.epochs": str(len(result.training_losses))}
        if result.training.best_epoch is not None:
            tags["training.best_epoch"] = str(result.training.best_epoch)
        self.tracker.set_tags(tags)

        metrics = dict(result.metrics)
        if result.training.best_validation_loss is not None:
            metrics["training.best_validation_loss"] = (
                result.training.best_validation_loss
            )
        if metrics:
            self.tracker.log_metrics(metrics)
        if self.model is not None and self.input_example is not None:
            self._log_model(self.model, self.input_example)
        super().on_run_finished(result)

    def _log_model(self, model: torch.nn.Module, input_example: np.ndarray) -> None:
        values = np.asarray(input_example, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError(
                "model input_example must have shape (channels, height, width)"
            )
        batch = np.expand_dims(values, axis=0)
        model_copy = deepcopy(model).to("cpu").eval()
        with torch.no_grad():
            output = model_copy(torch.from_numpy(batch)).detach().numpy()
        self.tracker.log_pytorch_model(
            model_copy,
            name=self.model_name,
            input_example=batch,
            output_example=output,
            registered_model_name=self.registered_model_name,
            metadata={"model_type": "reconstruction"},
        )
