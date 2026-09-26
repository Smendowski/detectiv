from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import pytest

from detectiv.tracking import MLflowTracker


class FakeRun:
    @property
    def info(self) -> object:
        return type(
            "Info", (), {"run_id": "native-run", "artifact_uri": "mlruns:/native-run"}
        )()


class FakeMLflow(ModuleType):
    def __init__(self) -> None:
        super().__init__("mlflow")
        self.experiment: str | None = None
        self.start_arguments: dict[str, object] | None = None
        self.parameters: dict[str, object] = {}
        self.dicts: list[tuple[Mapping[str, object], str]] = []
        self.metrics: list[tuple[str, float, int | None]] = []
        self.tags: dict[str, str] = {}
        self.artifacts: list[tuple[str, str]] = []
        self.end_status: str | None = None
        self.run = FakeRun()

    def set_experiment(self, name: str) -> None:
        self.experiment = name

    def start_run(self, **kwargs: object) -> FakeRun:
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

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self.tags.update(tags)

    def log_artifacts(self, directory: str, artifact_path: str) -> None:
        self.artifacts.append((directory, artifact_path))

    def end_run(self, *, status: str) -> None:
        self.end_status = status


@pytest.fixture
def mlflow(monkeypatch: pytest.MonkeyPatch) -> FakeMLflow:
    fake = FakeMLflow()
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake)
    return fake


def test_tracker_starts_and_publishes_to_mlflow(
    mlflow: FakeMLflow, tmp_path: Path
) -> None:
    tracker = MLflowTracker(
        run_name="text",
        parameters={"trainer": {"epochs": 3}, "seed": 7},
        configuration={"window": {"size": 32}},
        dataset={"name": "synthetic"},
        tags={"detectiv.python": "user"},
        description="Generic tracking.",
    )

    assert tracker.start() == ("native-run", "mlruns:/native-run")
    tracker.log_metrics({"test": {"score": 0.9, "label": "ignored"}})
    tracker.log_artifacts(tmp_path, artifact_path="results")
    tracker.close()

    assert mlflow.experiment == "experiments"
    assert mlflow.parameters == {"trainer.epochs": 3, "seed": 7}
    assert mlflow.dicts == [
        ({"window": {"size": 32}}, "detectiv/configuration.json"),
        ({"name": "synthetic"}, "detectiv/dataset.json"),
    ]
    assert mlflow.tags["detectiv.python"] != "user"
    assert mlflow.metrics == [("test.score", 0.9, None)]
    assert mlflow.artifacts == [(str(tmp_path), "results")]
    assert mlflow.end_status == "FINISHED"


def test_tracker_rejects_publication_outside_a_run() -> None:
    tracker = MLflowTracker("benchmark")

    with pytest.raises(RuntimeError, match="not active"):
        tracker.log_metrics({"score": 1.0})
    with pytest.raises(RuntimeError, match="not active"):
        tracker.log_artifacts(Path("missing"))


def test_tracker_requires_the_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_mlflow(_: str) -> ModuleType:
        raise ModuleNotFoundError()

    monkeypatch.setattr(importlib, "import_module", missing_mlflow)

    with pytest.raises(ModuleNotFoundError, match="uv sync --extra experiment"):
        MLflowTracker("benchmark").start()


def test_tracker_closes_a_run_when_setup_logging_fails(
    mlflow: FakeMLflow, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_: Mapping[str, object]) -> None:
        raise RuntimeError("unavailable tracking server")

    monkeypatch.setattr(mlflow, "log_params", fail)

    with pytest.raises(RuntimeError, match="unavailable tracking server"):
        MLflowTracker("benchmark", parameters={"seed": 42}).start()

    assert mlflow.end_status == "FAILED"
