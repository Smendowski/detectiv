from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import ExitStack
from typing import Any
from uuid import uuid4

from detectiv.callbacks.base import BaseCallback
from detectiv.reports import CompletedRunSummary, RunContext
from detectiv.reproducibility import ReproducibilitySettings


class BaseScenario[T](ABC):
    """Generic scenario orchestration with ordered callback lifecycle dispatch."""

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
        """Execute one scenario run and publish its completed output summary.

        Returns:
            The final result after ordered completion-callback enrichment.

        Raises:
            BaseException: Propagates scenario and callback lifecycle failures.
        """
        started: list[BaseCallback[T]] = []
        context = RunContext(str(uuid4()), self.scenario_type)
        self.completed_run = None

        try:
            if self.reproducibility is not None:
                self.reproducibility.apply()
            self._notify_started(started, context)
            result = self._run()
            result = self._notify_finished(started, result)
        except BaseException as error:
            self._notify_failed(started, error)
            raise
        finally:
            self._notify_closed(started)

        self.completed_run = context.completed_run
        return result

    @abstractmethod
    def _run(self) -> T:
        raise NotImplementedError

    def _notify_epoch_finished(self, event: Any) -> None:
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
    def _notify_finished(callbacks: Sequence[BaseCallback[T]], result: T) -> T:
        for callback in callbacks:
            replacement = callback.on_run_finished(result)
            if replacement is not None:
                result = replacement
        return result

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
        with ExitStack() as stack:
            for callback in callbacks:
                stack.callback(callback.on_run_closed)
