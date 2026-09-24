from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CompletedRunSummary:
    """Discoverable local and MLflow outputs of a completed Detectiv run."""

    run_id: str
    artifact_location: Path | str | None = None
    mlflow_run_id: str | None = None
    mlflow_location: Path | str | None = None


@dataclass
class RunContext:
    """Shared run identity and output registry provided to lifecycle callbacks."""

    run_id: str
    scenario_type: str | None = None
    _artifact_location: Path | str | None = field(default=None, init=False, repr=False)
    _mlflow_run_id: str | None = field(default=None, init=False, repr=False)
    _mlflow_location: Path | str | None = field(default=None, init=False, repr=False)

    def register_local_artifacts(self, location: Path) -> None:
        """Register a completed local artifact bundle."""
        self._artifact_location = location

    def register_mlflow(self, run_id: str, location: str | None = None) -> None:
        """Register native MLflow output metadata for this execution."""
        self._mlflow_run_id = run_id
        self._mlflow_location = location

    @property
    def completed_run(self) -> CompletedRunSummary:
        """Return the currently known immutable output summary."""
        return CompletedRunSummary(
            run_id=self.run_id,
            artifact_location=self._artifact_location,
            mlflow_run_id=self._mlflow_run_id,
            mlflow_location=self._mlflow_location,
        )
