from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from detectiv.runs import RunContext, TrainingEpochEvent


class Callback[T](Protocol):
    """Structural lifecycle contract used by scenario callback dispatch."""

    @property
    def name(self) -> str:
        """Return the callback's unique scenario registration name.

        Returns:
            The callback registration name.
        """
        ...

    def on_run_started(self) -> None:
        """Initialize callback state before scenario work begins."""
        ...

    def on_run_context(self, context: RunContext) -> None:
        """Receive the shared run context.

        Args:
            context: Run identity and output-registration context.
        """
        ...

    def on_epoch_finished(self, event: TrainingEpochEvent) -> None:
        """Handle one completed training epoch.

        Args:
            event: Data recorded for the completed epoch.
        """
        ...

    def on_run_finished(self, result: T) -> T | None:
        """Handle or replace a successful result.

        Args:
            result: Current result after preceding callbacks.

        Returns:
            A replacement result, or `None` to preserve the current result.
        """
        ...

    def on_run_failed(self, error: BaseException) -> None:
        """Handle a scenario failure.

        Args:
            error: Original scenario exception.
        """
        ...

    def on_run_closed(self) -> None:
        """Release callback resources after success or failure."""
        ...


class BaseCallback[T]:
    """Receives ordered lifecycle notifications from a scenario run.

    Subclasses must provide a unique, stable name within each scenario. A
    callback is started before work begins, receives completed training epochs,
    then receives either successful completion or failure notification.
    """

    @property
    def name(self) -> str:
        """Return the callback's unique scenario registration name.

        Returns:
            The callback registration name.

        Raises:
            NotImplementedError: If a subclass does not define a name.
        """
        raise NotImplementedError

    def on_run_started(self) -> None:
        """Initialize state after reproducibility settings are applied.

        Called in registration order before scenario work begins. If this hook
        raises, this callback is not considered started and will not receive a
        failure notification for that error.
        """
        pass

    def on_run_context(self, context: RunContext) -> None:
        """Receive the shared identity and output-registration context.

        This optional hook runs before ``on_run_started()``. Existing callbacks
        that implement only the original lifecycle hooks remain compatible.

        Args:
            context: Run identity and output-registration context.
        """
        pass

    def on_epoch_finished(self, event: TrainingEpochEvent) -> None:
        """Handle one completed training epoch in registration order.

        Args:
            event: Losses, learning rates, epoch index, and elapsed time for
                the completed epoch.

        Raises:
            BaseException: Propagated by the scenario, which then notifies each
                already-started callback of failure.
        """
        pass

    def on_run_finished(self, result: T) -> T | None:
        """Handle or replace the successful result in registration order.

        Returning ``None`` preserves ``result`` for the next callback. Exceptions
        propagate directly and do not trigger `on_run_failed()`.

        Args:
            result: Current result after preceding callbacks.

        Returns:
            A replacement result, or `None` to preserve the current result.
        """
        return None

    def on_run_failed(self, error: BaseException) -> None:
        """Handle failure after this callback has started.

        Args:
            error: The original exception from startup, execution, or an epoch
                hook.

        Exceptions from this hook are attached as notes to the original error
        when they are `Exception` instances; they never replace it.
        """
        pass

    def on_run_closed(self) -> None:
        """Release resources after success or failure notification.

        Called exactly once for each callback whose start hook completed.
        """
        pass
