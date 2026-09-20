import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from uuid import uuid4

from detectiv.callbacks.base import BaseCallback
from detectiv.runs import (
    CompletedRunSummary,
    ReproducibilitySettings,
    RunContext,
    RunIdentity,
    TrainingEpochEvent,
)


class BaseScenario[T](ABC):
    scenario_type: str | None = None

    def __init__(
        self,
        *,
        callbacks: Sequence[BaseCallback[T]] = (),
        reproducibility: ReproducibilitySettings | None = None,
    ) -> None:
        if len({callback.name for callback in callbacks}) != len(callbacks):
            raise ValueError("callback names must be unique")
        self.callbacks = tuple(callbacks)
        self.reproducibility = reproducibility
        self.completed_run: CompletedRunSummary | None = None

    def run(self) -> T:
        started: list[BaseCallback[T]] = []
        context = RunContext(RunIdentity(str(uuid4())), self.scenario_type)
        self.completed_run = None

        try:
            if self.reproducibility is not None:
                self.reproducibility.apply()
            self._notify_started(started, context)
            result = self._run()
        except BaseException as error:
            self._notify_failed(started, error)
            raise

        else:
            self._notify_finished(started, result)
            self.completed_run = context.completed_run
            return result
        finally:
            self._notify_closed(started)

    @abstractmethod
    def _run(self) -> T:
        raise NotImplementedError

    def _notify_epoch_finished(self, event: TrainingEpochEvent) -> None:
        for callback in self.callbacks:
            callback.on_epoch_finished(event)

    def _notify_started(
        self, started: list[BaseCallback[T]], context: RunContext
    ) -> None:
        for callback in self.callbacks:
            callback.on_run_context(context)
            callback.on_run_started()
            started.append(callback)

    @staticmethod
    def _notify_finished(callbacks: Sequence[BaseCallback[T]], result: T) -> None:
        for callback in callbacks:
            callback.on_run_finished(result)

    @staticmethod
    def _notify_failed(
        callbacks: Sequence[BaseCallback[T]], error: BaseException
    ) -> None:
        for callback in callbacks:
            try:
                callback.on_run_failed(error)
            except Exception as notification_error:
                error.add_note(
                    f"Callback {callback.name!r} failed while reporting the run "
                    f"failure: {notification_error!r}"
                )

    @staticmethod
    def _notify_closed(callbacks: Sequence[BaseCallback[T]]) -> None:
        original_error = sys.exception()
        for callback in reversed(callbacks):
            try:
                callback.on_run_closed()
            except Exception as cleanup_error:
                if original_error is None:
                    raise
                original_error.add_note(
                    f"Callback {callback.name!r} failed while closing the run: "
                    f"{cleanup_error!r}"
                )
