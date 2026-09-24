import random
from uuid import UUID

import pytest

from detectiv.callbacks import BaseCallback
from detectiv.reproducibility import ReproducibilitySettings
from detectiv.runs import RunContext
from detectiv.scenarios import BaseScenario


class RecordingCallback(BaseCallback[str]):
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        start_error: Exception | None = None,
        finish_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self._name = name
        self.events = events
        self.start_error = start_error
        self.finish_error = finish_error
        self.close_error = close_error

    @property
    def name(self) -> str:
        return self._name

    def on_run_started(self) -> None:
        self.events.append(f"{self.name}:started")
        if self.start_error is not None:
            raise self.start_error

    def on_run_finished(self, result: object) -> None:
        self.events.append(f"{self.name}:finished:{result}")
        if self.finish_error is not None:
            raise self.finish_error
        return None

    def on_run_failed(self, error: BaseException) -> None:
        self.events.append(f"{self.name}:failed:{type(error).__name__}")

    def on_run_closed(self) -> None:
        self.events.append(f"{self.name}:closed")
        if self.close_error is not None:
            raise self.close_error


class ExampleScenario(BaseScenario[str]):
    def __init__(
        self,
        *,
        callbacks: tuple[BaseCallback[str], ...] = (),
        error: BaseException | None = None,
        reproducibility: ReproducibilitySettings | None = None,
    ) -> None:
        super().__init__(callbacks=callbacks, reproducibility=reproducibility)
        self.error = error

    def _run(self) -> str:
        if self.error is not None:
            raise self.error
        return "result"


class ReplacingCallback(BaseCallback[str]):
    def __init__(self, name: str, events: list[str], suffix: str) -> None:
        self._name = name
        self.events = events
        self.suffix = suffix

    @property
    def name(self) -> str:
        return self._name

    def on_run_finished(self, result: str) -> str:
        self.events.append(f"{self.name}:finished:{result}")
        return f"{result}:{self.suffix}"


def test_base_scenario_dispatches_lifecycle_for_a_new_scenario_type() -> None:
    events: list[str] = []

    result = ExampleScenario(callbacks=(RecordingCallback("first", events),)).run()

    assert result == "result"
    assert events == ["first:started", "first:finished:result", "first:closed"]


def test_base_scenario_exposes_one_uuid_context_and_completed_summary() -> None:
    contexts: list[RunContext] = []

    class ContextCallback(BaseCallback[str]):
        @property
        def name(self) -> str:
            return "context"

        def on_run_context(self, context: RunContext) -> None:
            contexts.append(context)

    scenario = ExampleScenario(callbacks=(ContextCallback(),))

    scenario.run()

    assert len(contexts) == 1
    UUID(contexts[0].identity.run_id)
    assert scenario.completed_run is not None
    assert scenario.completed_run.run_id == contexts[0].identity.run_id
    assert scenario.completed_run.artifact_location is None
    assert scenario.completed_run.mlflow_run_id is None


def test_base_scenario_applies_reproducibility_before_start_callbacks() -> None:
    settings = ReproducibilitySettings(seed=42)
    observed: list[int] = []

    class SeedCallback(BaseCallback[str]):
        @property
        def name(self) -> str:
            return "seed"

        def on_run_started(self) -> None:
            observed.append(random.randrange(1_000_000))

    ExampleScenario(callbacks=(SeedCallback(),), reproducibility=settings).run()
    settings.apply()

    assert observed == [random.randrange(1_000_000)]


def test_base_scenario_preserves_failure_lifecycle_for_a_new_scenario_type() -> None:
    events: list[str] = []

    with pytest.raises(RuntimeError, match="failed"):
        ExampleScenario(
            callbacks=(RecordingCallback("first", events),),
            error=RuntimeError("failed"),
        ).run()

    assert events == ["first:started", "first:failed:RuntimeError", "first:closed"]


def test_base_scenario_notifies_started_callbacks_when_interrupted() -> None:
    events: list[str] = []

    with pytest.raises(KeyboardInterrupt):
        ExampleScenario(
            callbacks=(RecordingCallback("first", events),),
            error=KeyboardInterrupt(),
        ).run()

    assert events == ["first:started", "first:failed:KeyboardInterrupt", "first:closed"]


def test_base_scenario_preserves_callback_order_for_non_reconstruction_results() -> (
    None
):
    events: list[str] = []

    ExampleScenario(
        callbacks=(
            RecordingCallback("first", events),
            RecordingCallback("second", events),
        )
    ).run()

    assert events == [
        "first:started",
        "second:started",
        "first:finished:result",
        "second:finished:result",
        "second:closed",
        "first:closed",
    ]


def test_base_scenario_chains_replacements_and_returns_the_final_result() -> None:
    events: list[str] = []

    result = ExampleScenario(
        callbacks=(
            RecordingCallback("observer", events),
            ReplacingCallback("first", events, "one"),
            ReplacingCallback("second", events, "two"),
            RecordingCallback("final_observer", events),
        )
    ).run()

    assert result == "result:one:two"
    assert events == [
        "observer:started",
        "final_observer:started",
        "observer:finished:result",
        "first:finished:result",
        "second:finished:result:one",
        "final_observer:finished:result:one:two",
        "final_observer:closed",
        "observer:closed",
    ]


def test_completion_callback_failure_enters_failure_lifecycle() -> None:
    events: list[str] = []

    with pytest.raises(RuntimeError, match="finished"):
        ExampleScenario(
            callbacks=(
                RecordingCallback("first", events),
                RecordingCallback(
                    "failing", events, finish_error=RuntimeError("finished")
                ),
                RecordingCallback("later", events),
            )
        ).run()

    assert events == [
        "first:started",
        "failing:started",
        "later:started",
        "first:finished:result",
        "failing:finished:result",
        "first:failed:RuntimeError",
        "failing:failed:RuntimeError",
        "later:failed:RuntimeError",
        "later:closed",
        "failing:closed",
        "first:closed",
    ]


def test_started_callback_failure_notifies_only_previously_started_callbacks() -> None:
    events: list[str] = []

    with pytest.raises(RuntimeError, match="start"):
        ExampleScenario(
            callbacks=(
                RecordingCallback("first", events),
                RecordingCallback("failing", events, start_error=RuntimeError("start")),
                RecordingCallback("last", events),
            )
        ).run()

    assert events == [
        "first:started",
        "failing:started",
        "first:failed:RuntimeError",
        "first:closed",
    ]


def test_base_scenario_preserves_original_error_when_cleanup_fails() -> None:
    events: list[str] = []

    with pytest.raises(RuntimeError, match="failed") as raised:
        ExampleScenario(
            callbacks=(
                RecordingCallback("first", events, close_error=RuntimeError("close")),
            ),
            error=RuntimeError("failed"),
        ).run()

    assert events == ["first:started", "first:failed:RuntimeError", "first:closed"]
    assert "failed while closing" in "\n".join(raised.value.__notes__)
