from time import perf_counter

from detectiv.callbacks.base import BaseCallback


class TimingCallback(BaseCallback[object]):
    """Measure elapsed wall-clock time between run start and terminal hook.

    `elapsed_seconds` is `None` until a started run finishes or fails. The
    value excludes reproducibility setup and includes callback-dispatched
    scenario work until this callback's terminal hook is called.
    """

    def __init__(self) -> None:
        """Initialize an idle timer with no elapsed duration."""
        self.elapsed_seconds: float | None = None
        self._started_at: float | None = None

    @property
    def name(self) -> str:
        """Return the fixed registration name `timing`.

        Returns:
            The callback registration name.
        """
        return "timing"

    def on_run_started(self) -> None:
        """Reset prior timing and start a new monotonic timer."""
        self.elapsed_seconds = None
        self._started_at = perf_counter()

    def on_run_finished(self, result: object) -> None:
        """Stop the timer after successful execution.

        Args:
            result: Successful scenario result; unused by this observer.
        """
        self._finish()
        return None

    def on_run_failed(self, error: BaseException) -> None:
        """Stop the timer after failure.

        Args:
            error: Scenario error; neither suppressed nor changed.
        """
        self._finish()

    def _finish(self) -> None:
        started_at = self._started_at
        if started_at is None:
            return
        self._started_at = None
        self.elapsed_seconds = perf_counter() - started_at
