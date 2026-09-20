import importlib
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Literal

import pytest

from detectiv.callbacks import BaseCallback, MlflowCallback
from detectiv.scenarios import BaseScenario


class FakeRunContext:
    def __init__(self) -> None:
        self.exit_arguments: tuple[object, ...] | None = None

    def __enter__(self) -> "FakeRunContext":
        return self

    def __exit__(self, *args: object) -> Literal[False]:
        self.exit_arguments = args
        return False

    @property
    def info(self) -> object:
        return type(
            "Info", (), {"run_id": "native-run", "artifact_uri": "mlruns:/native-run"}
        )()


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

    def active_run(self) -> FakeRunContext:
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


@pytest.fixture
def mlflow(monkeypatch: pytest.MonkeyPatch) -> FakeMlflow:
    fake = FakeMlflow()
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake)
    return fake


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


def test_generic_tracker_supports_a_non_reconstruction_scenario(
    mlflow: FakeMlflow, tmp_path: Path
) -> None:
    (tmp_path / "result.txt").write_text("result")
    tracking: MlflowCallback[str] = MlflowCallback(
        run_name="text",
        parameters={"trainer": {"epochs": 3}, "seed": 7},
        configuration={"window": {"size": 32}},
        dataset={"name": "synthetic"},
        tags={"detectiv.run_status": "user", "detectiv.python": "user"},
        description="Generic tracking.",
        metrics_provider=lambda: {"test": {"score": 0.9, "label": "ignored"}},
        artifact_directories={tmp_path: "results"},
        tracking_uri="http://mlflow:5000",
    )
    scenario = TextScenario(callbacks=(tracking,))

    assert scenario.run() == "complete"

    assert mlflow.tracking_uri == "http://mlflow:5000"
    assert mlflow.experiment == "experiments"
    assert mlflow.parameters == {"trainer.epochs": 3, "seed": 7}
    assert mlflow.dicts == [
        ({"window": {"size": 32}}, "detectiv/configuration.json"),
        ({"name": "synthetic"}, "detectiv/dataset.json"),
    ]
    assert mlflow.tags["detectiv.run_status"] == "succeeded"
    assert mlflow.tags["detectiv.scenario_type"] == "text"
    assert mlflow.tags["detectiv.python"] != "user"
    assert mlflow.metrics == [("test.score", 0.9, None)]
    assert mlflow.artifacts == [(str(tmp_path), "results")]
    assert scenario.completed_run is not None
    assert scenario.completed_run.mlflow_run_id == "native-run"
    assert scenario.completed_run.mlflow_location == "mlruns:/native-run"


def test_generic_tracker_records_failure_lifecycle(mlflow: FakeMlflow) -> None:
    scenario = TextScenario(
        callbacks=(MlflowCallback("benchmark"),), error=ValueError("bad input")
    )

    with pytest.raises(ValueError, match="bad input"):
        scenario.run()

    assert mlflow.tags["detectiv.run_status"] == "failed"
    assert mlflow.tags["detectiv.error_type"] == "ValueError"
    assert mlflow.run.exit_arguments is not None
    assert isinstance(mlflow.run.exit_arguments[1], ValueError)


def test_generic_tracker_rejects_publication_outside_a_run() -> None:
    tracking: MlflowCallback[str] = MlflowCallback("benchmark")

    with pytest.raises(RuntimeError, match="not active"):
        tracking.log_metrics({"score": 1.0})
    with pytest.raises(RuntimeError, match="not active"):
        tracking.log_artifacts(Path("missing"))


def test_tracking_requires_the_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_mlflow(_: str) -> ModuleType:
        raise ModuleNotFoundError()

    monkeypatch.setattr(importlib, "import_module", missing_mlflow)

    with pytest.raises(ModuleNotFoundError, match="uv sync --extra experiment"):
        TextScenario(callbacks=(MlflowCallback("benchmark"),)).run()


def test_untracked_scenarios_do_not_import_mlflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = importlib.import_module

    def reject_mlflow(name: str, package: str | None = None) -> ModuleType:
        if name == "mlflow":
            raise AssertionError("untracked scenarios must not import MLflow")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", reject_mlflow)

    assert TextScenario(callbacks=()).run() == "complete"


def test_tracker_closes_a_run_when_setup_logging_fails(
    mlflow: FakeMlflow, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_: Mapping[str, object]) -> None:
        raise RuntimeError("unavailable tracking server")

    monkeypatch.setattr(mlflow, "log_params", fail)

    with pytest.raises(RuntimeError, match="unavailable tracking server"):
        TextScenario(
            callbacks=(MlflowCallback("benchmark", parameters={"seed": 42}),)
        ).run()

    assert mlflow.run.exit_arguments is not None
    assert isinstance(mlflow.run.exit_arguments[1], RuntimeError)
