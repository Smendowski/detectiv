from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import torch

from detectiv.callbacks import BaseCallback, BaseMLflowCallback, MLflowCallback
from detectiv.models.autoencoders import TrainingEpochEvent, TrainingHistory
from detectiv.reports import ReconstructionReport, RunContext
from detectiv.scenarios import BaseScenario
from detectiv.tracking import MLflowTracker


class RecordingTracker(MLflowTracker):
    def __init__(self) -> None:
        super().__init__()
        self.is_active = False
        self.tags: dict[str, str] = {}
        self.metrics: list[tuple[dict[str, object], int | None]] = []
        self.artifacts: list[tuple[Path, str]] = []
        self.closed_with: list[BaseException | None] = []
        self.models: list[dict[str, object]] = []

    @property
    def active(self) -> bool:
        return self.is_active

    def start(self) -> tuple[str, str | None]:
        self.is_active = True
        return "native-run", "mlruns:/native-run"

    def close(self, error: BaseException | None = None) -> None:
        if self.is_active:
            self.closed_with.append(error)
            self.is_active = False

    def log_metrics(
        self, metrics: Mapping[str, object], *, step: int | None = None
    ) -> None:
        self.metrics.append((dict(metrics), step))

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self.tags.update(tags)

    def log_artifacts(
        self, directory: Path, *, artifact_path: str = "detectiv"
    ) -> None:
        self.artifacts.append((directory, artifact_path))

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
        self.models.append(
            {
                "model": model,
                "name": name,
                "input_example": input_example,
                "output_example": output_example,
                "registered_model_name": registered_model_name,
                "metadata": dict(metadata or {}),
            }
        )


class TextScenario(BaseScenario[str]):
    scenario_type = "text"

    def __init__(
        self,
        *,
        callbacks: tuple[BaseCallback[str], ...],
        error: Exception | None = None,
    ) -> None:
        super().__init__(callbacks=callbacks)
        self.error = error

    def _run(self) -> str:
        if self.error is not None:
            raise self.error
        return "complete"


def test_base_mlflow_callback_adapts_a_successful_run(tmp_path: Path) -> None:
    tracker = RecordingTracker()
    callback = BaseMLflowCallback[str](
        tracker,
        metrics_provider=lambda result: {"test.score": float(len(result))},
        artifact_directories={tmp_path: "results"},
    )
    scenario = TextScenario(callbacks=(callback,))

    assert scenario.run() == "complete"

    assert tracker.tags["detectiv.run_status"] == "succeeded"
    assert tracker.tags["detectiv.scenario_type"] == "text"
    assert tracker.metrics == [({"test.score": 8.0}, None)]
    assert tracker.artifacts == [(tmp_path, "results")]
    assert tracker.closed_with == [None]
    assert scenario.completed_run is not None
    assert scenario.completed_run.mlflow_run_id == "native-run"
    assert scenario.completed_run.mlflow_location == "mlruns:/native-run"


def test_base_mlflow_callback_records_failure() -> None:
    tracker = RecordingTracker()
    scenario = TextScenario(
        callbacks=(BaseMLflowCallback[str](tracker),),
        error=ValueError("bad input"),
    )

    with pytest.raises(ValueError, match="bad input"):
        scenario.run()

    assert tracker.tags["detectiv.run_status"] == "failed"
    assert tracker.tags["detectiv.error_type"] == "ValueError"
    assert isinstance(tracker.closed_with[0], ValueError)


def test_base_mlflow_callback_records_a_later_callback_failure() -> None:
    class FailingPublisher(BaseCallback[str]):
        @property
        def name(self) -> str:
            return "publisher"

        def on_run_finished(self, result: str) -> None:
            raise RuntimeError("publication failed")

    tracker = RecordingTracker()
    scenario = TextScenario(
        callbacks=(BaseMLflowCallback[str](tracker), FailingPublisher())
    )

    with pytest.raises(RuntimeError, match="publication failed"):
        scenario.run()

    assert tracker.tags["detectiv.run_status"] == "failed"
    assert isinstance(tracker.closed_with[0], RuntimeError)


def test_base_mlflow_callback_can_be_reused_after_failure() -> None:
    tracker = RecordingTracker()
    callback = BaseMLflowCallback[str](tracker)

    with pytest.raises(ValueError):
        TextScenario(callbacks=(callback,), error=ValueError("first run")).run()
    assert TextScenario(callbacks=(callback,)).run() == "complete"

    assert isinstance(tracker.closed_with[0], ValueError)
    assert tracker.closed_with[1] is None
    assert tracker.tags["detectiv.run_status"] == "succeeded"


def test_mlflow_callback_logs_reconstruction_training_and_metrics() -> None:
    tracker = RecordingTracker()
    callback = MLflowCallback(tracker)
    callback.on_run_context(_context())
    callback.on_run_started()
    callback.on_epoch_finished(TrainingEpochEvent(0, 0.5, 0.4, (1e-3,), 2.0))
    callback.on_run_finished(_result().with_metrics({"evaluation.score": 0.9}))
    callback.on_run_closed()

    assert tracker.metrics == [
        (
            {
                "training.loss": 0.5,
                "validation.loss": 0.4,
                "training.learning_rate.0": 1e-3,
                "training.epoch_seconds": 2.0,
            },
            0,
        ),
        (
            {
                "evaluation.score": 0.9,
                "training.best_validation_loss": 0.4,
            },
            None,
        ),
    ]
    assert tracker.tags["training.epochs"] == "1"
    assert tracker.tags["training.best_epoch"] == "0"
    assert tracker.tags["detectiv.run_status"] == "succeeded"


def test_mlflow_callback_logs_the_reconstruction_model() -> None:
    tracker = RecordingTracker()
    callback = MLflowCallback(
        tracker,
        model=torch.nn.Identity(),
        input_example=np.zeros((1, 2, 2)),
    )
    callback.on_run_context(_context())
    callback.on_run_started()
    callback.on_run_finished(_result())
    callback.on_run_closed()

    assert len(tracker.models) == 1
    assert tracker.models[0]["name"] == "reconstruction_model"
    assert tracker.models[0]["metadata"] == {"model_type": "reconstruction"}


def test_mlflow_callback_requires_complete_model_configuration() -> None:
    with pytest.raises(ValueError, match="model and input_example"):
        MLflowCallback(RecordingTracker(), model=torch.nn.Identity())


def _context() -> RunContext:
    return RunContext("run", "reconstruction")


def _result() -> ReconstructionReport:
    return ReconstructionReport(
        window_scores={},
        point_scores={},
        training=TrainingHistory(
            (0.5,), (0.4,), best_epoch=0, best_validation_loss=0.4
        ),
    )
