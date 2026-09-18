from time import perf_counter
from typing import TYPE_CHECKING

from detectiv.callbacks.base import ReconstructionCallback

if TYPE_CHECKING:
    from detectiv.scenarios.reconstruction import ReconstructionScenarioResult


class TimingCallback(ReconstructionCallback):
    def __init__(self) -> None:
        self.elapsed_seconds: float | None = None
        self._started_at: float | None = None

    @property
    def name(self) -> str:
        return "timing"

    def on_run_started(self) -> None:
        self.elapsed_seconds = None
        self._started_at = perf_counter()

    def on_run_finished(self, result: "ReconstructionScenarioResult") -> None:
        self._finish()

    def on_run_failed(self, error: BaseException) -> None:
        self._finish()

    def _finish(self) -> None:
        started_at = self._started_at
        if started_at is None:
            return
        self._started_at = None
        self.elapsed_seconds = perf_counter() - started_at
