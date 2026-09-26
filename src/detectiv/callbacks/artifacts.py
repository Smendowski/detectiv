from collections.abc import Mapping
from pathlib import Path

from detectiv.callbacks.base import BaseCallback
from detectiv.reports import (
    ReconstructionReport,
    ReconstructionReportWriter,
    ReportArtifacts,
    RunContext,
)


class ReportArtifactCallback(BaseCallback[ReconstructionReport]):
    """Publish a local report bundle only after a successful run.

    Args:
        directory: Explicit destination, or ``None`` for ``artifacts/<run_id>``.
        provenance: Metadata recorded with the bundle.
        visualize: Whether the writer should render visual artifacts.
        overwrite: Whether the writer may replace an existing destination.

    The completed bundle is exposed through `artifacts`; failures leave no new
    completed bundle and preserve writer errors.
    """

    def __init__(
        self,
        directory: Path | str | None = None,
        *,
        provenance: Mapping[str, object] | None = None,
        visualize: bool = False,
        overwrite: bool = False,
    ) -> None:
        """Configure the destination report writer."""
        self.directory = None if directory is None else Path(directory)
        self.provenance = provenance
        self.visualize = visualize
        self.overwrite = overwrite
        self.artifacts: ReportArtifacts | None = None
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
        """Retain the context used to derive and register artifact locations.

        Args:
            context: Shared run identity and output-registration context.
        """
        self._context = context

    def on_run_finished(self, result: ReconstructionReport) -> None:
        """Write `result` and store its completed artifact bundle.

        Writer errors propagate, so no result is assigned to `artifacts`.

        Args:
            result: Completed result to serialize.

        Raises:
            RuntimeError: If the callback did not receive a run context.
        """
        if self._context is None:
            raise RuntimeError("artifact callback did not receive a run context")
        writer = ReconstructionReportWriter(
            self.directory or Path("artifacts") / self._context.run_id,
            provenance=self.provenance,
            visualize=self.visualize,
            overwrite=self.overwrite,
        )
        self.artifacts = writer.write(
            result,
            completed_run=self._context.completed_run,
        )
        self._context.register_local_artifacts(self.artifacts.manifest.parent)

    def on_run_closed(self) -> None:
        """Release the per-run context while preserving published artifacts."""
        self._context = None
