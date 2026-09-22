from collections.abc import Callable, Mapping
from pathlib import Path

from detectiv.callbacks.base import BaseCallback
from detectiv.runs import (
    RunArtifactResult,
    RunArtifacts,
    RunArtifactWriter,
    RunContext,
)


class RunArtifactCallback(BaseCallback[RunArtifactResult]):
    """Publish a local artifact bundle only after a successful run.

    Args:
        directory: Explicit destination, or ``None`` for ``artifacts/<run_id>``.
        metrics_provider: Optional zero-argument provider evaluated when the
            run succeeds.
        provenance: Metadata recorded with the bundle.
        visualize: Whether the writer should render visual artifacts.
        overwrite: Whether the writer may replace an existing destination.

    The completed bundle is exposed through `artifacts`; failures leave no new
    completed bundle and preserve writer errors.
    """

    def __init__(
        self,
        directory: Path | None = None,
        *,
        metrics_provider: Callable[[], Mapping[str, object]] | None = None,
        provenance: Mapping[str, object] | None = None,
        visualize: bool = False,
        overwrite: bool = False,
    ) -> None:
        """Configure the destination writer and optional completion metrics."""
        self.directory = directory
        self.provenance = provenance
        self.visualize = visualize
        self.overwrite = overwrite
        self.writer = (
            None
            if directory is None
            else RunArtifactWriter(
                directory,
                provenance=provenance,
                visualize=visualize,
                overwrite=overwrite,
            )
        )
        self.metrics_provider = metrics_provider
        self.artifacts: RunArtifacts | None = None
        self._context: RunContext | None = None

    @property
    def name(self) -> str:
        """Return the fixed registration name `artifacts`.

        Returns:
            The callback registration name.
        """
        return "artifacts"

    def on_run_started(self) -> None:
        """Clear the artifact reference before the run begins."""
        self.artifacts = None

    def on_run_context(self, context: RunContext) -> None:
        """Prepare the configured or run-ID-derived artifact destination.

        Args:
            context: Shared run identity and output-registration context.
        """
        self._context = context
        self.writer = RunArtifactWriter(
            self.directory or Path("artifacts") / context.identity.run_id,
            provenance=self.provenance,
            visualize=self.visualize,
            overwrite=self.overwrite,
        )

    def on_run_finished(self, result: RunArtifactResult) -> None:
        """Write `result` and store its completed artifact bundle.

        The metrics provider, when configured, is called once here. Writer and
        provider errors propagate, so no result is assigned to `artifacts`.

        Args:
            result: Completed result to serialize.

        Raises:
            RuntimeError: If the callback did not receive a run context.
        """
        if self.writer is None:
            raise RuntimeError("artifact callback did not receive a run context")
        self.artifacts = self.writer.write(
            result,
            metrics=None if self.metrics_provider is None else self.metrics_provider(),
            completed_run=(
                None if self._context is None else self._context.completed_run
            ),
        )
        if self._context is not None:
            self._context.register_local_artifacts(self.artifacts.manifest.parent)
        return None
