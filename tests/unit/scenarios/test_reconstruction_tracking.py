from collections.abc import Mapping
from types import ModuleType
from typing import Literal

import numpy as np
import pytest
import torch

from detectiv.callbacks import MlflowCallback
from detectiv.models.autoencoders import TrainingHistory
from detectiv.runs import RunContext, RunIdentity, TrainingEpochEvent
from detectiv.scenarios import (
    ReconstructionMlflowCallback,
    ReconstructionMlflowModelLogging,
    ReconstructionScenarioResult,
)


class FakeRunContext:
    def __enter__(self) -> "FakeRunContext":
        return self

    def __exit__(self, *args: object) -> Literal[False]:
        return False

    @property
    def info(self) -> object:
        return type("Info", (), {"run_id": "native-run", "artifact_uri": None})()


class FakeMlflow(ModuleType):
    def __init__(self) -> None:
        super().__init__("mlflow")
        self.metrics: list[tuple[str, float, int]] = []
        self.tags: dict[str, str] = {}
        self.artifacts: list[tuple[str, str]] = []
        self.figures: list[str] = []
        self.run = FakeRunContext()

    def set_experiment(self, name: str) -> None:
        pass

    def start_run(self, **kwargs: object) -> FakeRunContext:
        return self.run

    def active_run(self) -> FakeRunContext:
        return self.run

    def log_params(self, parameters: Mapping[str, object]) -> None:
        pass

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        self.metrics.extend((name, value, step) for name, value in metrics.items())

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        self.metrics.append((name, value, step or 0))

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self.tags.update(tags)

    def log_artifacts(self, directory: str, artifact_path: str) -> None:
        self.artifacts.append((directory, artifact_path))

    def log_figure(self, figure: object, artifact_file: str) -> None:
        self.figures.append(artifact_file)


@pytest.fixture
def mlflow(monkeypatch: pytest.MonkeyPatch) -> FakeMlflow:
    fake = FakeMlflow()
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake)
    return fake


def _result() -> ReconstructionScenarioResult:
    return ReconstructionScenarioResult(
        window_scores={},
        point_scores={},
        training=TrainingHistory(
            (0.5,), (0.4,), best_epoch=0, best_validation_loss=0.4
        ),
    )


def test_reconstruction_callbacks_compose_explicitly(mlflow: FakeMlflow) -> None:
    tracking: MlflowCallback[ReconstructionScenarioResult] = MlflowCallback("benchmark")
    callbacks = (
        tracking,
        ReconstructionMlflowCallback(tracking, log_training_curve=False),
    )

    assert [callback.name for callback in callbacks] == [
        "mlflow",
        "mlflow_reconstruction",
    ]
    tracking.on_run_context(RunContext(RunIdentity("run"), "reconstruction"))
    tracking.on_run_started()
    callbacks[1].on_epoch_finished(TrainingEpochEvent(0, 0.5, 0.4, (1e-3,), 2.0))
    callbacks[1].on_run_finished(_result())
    tracking.on_run_closed()

    assert mlflow.metrics == [
        ("training.loss", 0.5, 0),
        ("validation.loss", 0.4, 0),
        ("training.learning_rate.0", 1e-3, 0),
        ("training.epoch_seconds", 2.0, 0),
    ]
    assert mlflow.tags["training.epochs"] == "1"
    assert mlflow.tags["detectiv.scenario_type"] == "reconstruction"


def test_generic_tracker_accepts_a_reconstruction_result_without_extensions(
    mlflow: FakeMlflow,
) -> None:
    tracking: MlflowCallback[ReconstructionScenarioResult] = MlflowCallback("benchmark")
    tracking.on_run_context(RunContext(RunIdentity("run"), "reconstruction"))
    tracking.on_run_started()
    tracking.on_run_finished(_result())
    tracking.on_run_closed()

    assert mlflow.tags["detectiv.run_status"] == "succeeded"
    assert not any(name.startswith("training.") for name in mlflow.tags)
    assert mlflow.metrics == []


def test_reconstruction_extension_logs_a_model_with_its_semantic_type(
    mlflow: FakeMlflow, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = ModuleType("mlflow.models")
    models.infer_signature = lambda inputs, outputs: "signature"  # type: ignore[attr-defined]
    pytorch = ModuleType("mlflow.pytorch")
    logged: dict[str, object] = {}
    pytorch.log_model = lambda model, **kwargs: logged.update(kwargs)  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "mlflow.models", models)
    monkeypatch.setitem(__import__("sys").modules, "mlflow.pytorch", pytorch)
    tracking: MlflowCallback[ReconstructionScenarioResult] = MlflowCallback("benchmark")
    callback = ReconstructionMlflowCallback(
        tracking,
        log_training_curve=False,
        model_logging=ReconstructionMlflowModelLogging(),
        model=torch.nn.Identity(),
        input_example=np.zeros((1, 2, 2)),
    )

    tracking.on_run_context(RunContext(RunIdentity("run"), "reconstruction"))
    tracking.on_run_started()
    callback.on_run_finished(_result())
    tracking.on_run_closed()

    assert logged["model_type"] == "reconstruction"


def test_reconstruction_extension_logs_curve_and_report_bundle(
    mlflow: FakeMlflow, monkeypatch: pytest.MonkeyPatch
) -> None:
    pyplot = ModuleType("matplotlib.pyplot")
    figure = type("Figure", (), {"tight_layout": lambda self: None})()
    axis = type(
        "Axis",
        (),
        {
            "plot": lambda self, *args, **kwargs: None,
            "set": lambda self, **kwargs: None,
            "legend": lambda self: None,
        },
    )()
    pyplot.subplots = lambda: (figure, axis)  # type: ignore[attr-defined]
    pyplot.close = lambda value: None  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "matplotlib.pyplot", pyplot)
    tracking: MlflowCallback[ReconstructionScenarioResult] = MlflowCallback("benchmark")
    callback = ReconstructionMlflowCallback(
        tracking, report_metrics_provider=lambda: {"test": {"score": 0.9}}
    )

    tracking.on_run_context(RunContext(RunIdentity("run"), "reconstruction"))
    tracking.on_run_started()
    callback.on_run_finished(_result())
    tracking.on_run_closed()

    assert mlflow.figures == ["detectiv/training_loss.png"]
    assert len(mlflow.artifacts) == 1
    assert mlflow.artifacts[0][1] == "detectiv"


def test_reconstruction_extension_requires_complete_model_configuration() -> None:
    with pytest.raises(ValueError, match="model and input_example"):
        ReconstructionMlflowCallback(
            MlflowCallback("benchmark"),
            model_logging=ReconstructionMlflowModelLogging(),
        )
