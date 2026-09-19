import importlib
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Literal

import numpy as np
import pytest

from detectiv.callbacks import MlflowCallback, MlflowExperiment
from detectiv.models import TrainingEpochEvent
from detectiv.models.autoencoders import TrainingHistory
from detectiv.scenarios import ReconstructionScenarioResult


class FakeRunContext:
    def __init__(self) -> None:
        self.entered = False
        self.exit_arguments: tuple[object, ...] | None = None

    def __enter__(self) -> "FakeRunContext":
        self.entered = True
        return self

    def __exit__(self, *args: object) -> Literal[False]:
        self.exit_arguments = args
        return False


class FakeMlflow(ModuleType):
    def __init__(self) -> None:
        super().__init__("mlflow")
        self.tracking_uri: str | None = None
        self.experiment: str | None = None
        self.start_arguments: dict[str, object] | None = None
        self.parameters: dict[str, object] = {}
        self.dicts: list[tuple[Mapping[str, object], str]] = []
        self.metrics: list[tuple[str, float, int | None]] = []
        self.tags: dict[str, str] = {}
        self.artifacts: list[tuple[str, str]] = []
        self.figures: list[str] = []
        self.run = FakeRunContext()

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment = name

    def start_run(self, **kwargs: object) -> FakeRunContext:
        self.start_arguments = kwargs
        tags = kwargs.get("tags")
        if isinstance(tags, Mapping):
            self.tags.update({str(name): str(value) for name, value in tags.items()})
        return self.run

    def log_params(self, parameters: Mapping[str, object]) -> None:
        self.parameters.update(parameters)

    def log_dict(self, values: Mapping[str, object], path: str) -> None:
        self.dicts.append((values, path))

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        self.metrics.append((name, value, step))

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        self.metrics.extend((name, value, step) for name, value in metrics.items())

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


def test_experiment_logs_parameters_configuration_and_artifacts(
    mlflow: FakeMlflow, tmp_path: Path
) -> None:
    artifact = tmp_path / "result.txt"
    artifact.write_text("result")

    with MlflowExperiment(
        "benchmark",
        run_name="MSL5",
        parameters={"trainer": {"epochs": 3}, "seed": 7},
        configuration={"window": {"size": 32}},
        dataset={"name": "synthetic", "series": ["MSL5"]},
        tags={"series": "MSL5"},
        description="Validation of the temporal protocol.",
        tracking_uri="http://mlflow:5000",
    ) as experiment:
        assert isinstance(experiment.callback, MlflowCallback)
        experiment.log_artifacts(tmp_path)

    assert mlflow.tracking_uri == "http://mlflow:5000"
    assert mlflow.experiment == "benchmark"
    assert mlflow.start_arguments is not None
    assert mlflow.start_arguments["run_name"] == "MSL5"
    assert mlflow.start_arguments["log_system_metrics"] is True
    assert mlflow.parameters == {"trainer.epochs": 3, "seed": 7}
    assert mlflow.dicts == [
        ({"window": {"size": 32}}, "detectiv/configuration.json"),
        ({"name": "synthetic", "series": ["MSL5"]}, "detectiv/dataset.json"),
    ]
    assert mlflow.tags["mlflow.note.content"] == "Validation of the temporal protocol."
    assert len(mlflow.tags["detectiv.dataset.manifest_sha256"]) == 64
    assert mlflow.artifacts == [(str(tmp_path), "detectiv")]
    assert mlflow.run.exit_arguments == (None, None, None)


def test_callback_logs_training_and_validation_curves(mlflow: FakeMlflow) -> None:
    result = ReconstructionScenarioResult(
        window_scores={},
        point_scores={"plan": {"mean": {"series": np.array([1.0])}}},
        training=TrainingHistory(
            training_losses=(0.5, 0.25),
            validation_losses=(0.4, 0.2),
            best_epoch=1,
            best_validation_loss=0.2,
        ),
    )

    with MlflowExperiment(
        "benchmark",
        metrics_provider=lambda: {"test": {"auroc": 0.9}},
        log_training_curve=False,
    ) as experiment:
        experiment.callback.on_run_started()
        experiment.callback.on_epoch_finished(
            TrainingEpochEvent(0, 0.5, 0.4, (1e-3,), 2.0)
        )
        experiment.callback.on_epoch_finished(
            TrainingEpochEvent(1, 0.25, 0.2, (1e-3,), 3.0)
        )
        experiment.callback.on_run_finished(result)

    assert mlflow.metrics == [
        ("training.loss", 0.5, 0),
        ("validation.loss", 0.4, 0),
        ("training.learning_rate.0", 1e-3, 0),
        ("training.epoch_seconds", 2.0, 0),
        ("training.loss", 0.25, 1),
        ("validation.loss", 0.2, 1),
        ("training.learning_rate.0", 1e-3, 1),
        ("training.epoch_seconds", 3.0, 1),
        ("test.auroc", 0.9, None),
    ]
    assert mlflow.tags["training.epochs"] == "2"
    assert mlflow.tags["training.best_epoch"] == "1"
    assert mlflow.tags["training.best_validation_loss"] == "0.2"


def test_callback_logs_a_combined_training_loss_figure(mlflow: FakeMlflow) -> None:
    result = ReconstructionScenarioResult(
        window_scores={},
        point_scores={"plan": {"mean": {"series": np.array([1.0])}}},
        training=TrainingHistory(
            training_losses=(0.5, 0.25),
            validation_losses=(0.4, 0.2),
            best_epoch=1,
            best_validation_loss=0.2,
        ),
    )

    with MlflowExperiment("benchmark") as experiment:
        experiment.callback.on_run_finished(result)

    assert mlflow.figures == ["detectiv/training_loss.png"]


def test_callback_logs_a_report_bundle_when_configured(
    mlflow: FakeMlflow,
) -> None:
    result = ReconstructionScenarioResult(
        window_scores={},
        point_scores={"plan": {"mean": {"series": np.array([1.0])}}},
        training=TrainingHistory(training_losses=(0.5,)),
    )
    experiment = MlflowExperiment(
        "benchmark",
        report_metrics_provider=lambda: {"test": {"auroc": 0.9}},
        log_training_curve=False,
    )

    with experiment:
        experiment.callback.on_run_finished(result)

    assert len(mlflow.artifacts) == 1
    assert mlflow.artifacts[0][1] == "detectiv"


def test_experiment_requires_the_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_mlflow(_: str) -> ModuleType:
        raise ModuleNotFoundError()

    monkeypatch.setattr(importlib, "import_module", missing_mlflow)

    with (
        pytest.raises(ModuleNotFoundError, match="uv sync --extra experiment"),
        MlflowExperiment("benchmark"),
    ):
        pass


def test_experiment_closes_a_run_when_setup_logging_fails(
    mlflow: FakeMlflow, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_: Mapping[str, object]) -> None:
        raise RuntimeError("unavailable tracking server")

    monkeypatch.setattr(mlflow, "log_params", fail)

    with pytest.raises(RuntimeError, match="unavailable tracking server"):
        MlflowExperiment("benchmark", parameters={"seed": 42}).__enter__()

    assert mlflow.run.exit_arguments is not None
    assert isinstance(mlflow.run.exit_arguments[1], RuntimeError)
