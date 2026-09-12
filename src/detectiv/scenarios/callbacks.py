from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from detectiv.scenarios.reconstruction import ReconstructionScenarioResult


class ScenarioCallback:
    def on_run_started(self) -> None:
        pass

    def on_epoch_finished(self, epoch: int, loss: float) -> None:
        pass

    def on_run_finished(self, result: "ReconstructionScenarioResult") -> None:
        pass

    def on_run_failed(self, error: BaseException) -> None:
        pass


class TimeCallback(ScenarioCallback):
    def __init__(self) -> None:
        self.elapsed_seconds: float | None = None
        self._started_at: float | None = None

    def on_run_started(self) -> None:
        self.elapsed_seconds = None
        self._started_at = perf_counter()

    def on_run_finished(self, result: "ReconstructionScenarioResult") -> None:
        self._finish()

    def on_run_failed(self, error: BaseException) -> None:
        self._finish()

    def _finish(self) -> None:
        if self._started_at is not None:
            self.elapsed_seconds = perf_counter() - self._started_at
