from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RunIdentity:
    """Framework-neutral identity assigned to one Detectiv scenario execution."""

    run_id: str


@dataclass(frozen=True)
class RunOutputLocator:
    """Immutable locator for an optional output produced by a run."""

    location: Path | str | None = None
    native_run_id: str | None = None


@dataclass(frozen=True)
class CompletedRunSummary:
    """Discoverable local and MLflow outputs of a completed Detectiv run."""

    identity: RunIdentity
    local_artifacts: RunOutputLocator | None = None
    mlflow: RunOutputLocator | None = None

    @property
    def run_id(self) -> str:
        """Return the Detectiv run ID."""
        return self.identity.run_id

    @property
    def artifact_location(self) -> Path | str | None:
        """Return the local artifact location, when publication was enabled."""
        return None if self.local_artifacts is None else self.local_artifacts.location

    @property
    def mlflow_run_id(self) -> str | None:
        """Return the native MLflow run ID, when tracking was enabled."""
        return None if self.mlflow is None else self.mlflow.native_run_id

    @property
    def mlflow_location(self) -> Path | str | None:
        """Return the MLflow artifact URI or run location, when available."""
        return None if self.mlflow is None else self.mlflow.location


@dataclass
class RunContext:
    """Shared run identity and output registry provided to lifecycle callbacks."""

    identity: RunIdentity
    scenario_type: str | None = None
    _local_artifacts: RunOutputLocator | None = field(
        default=None, init=False, repr=False
    )
    _mlflow: RunOutputLocator | None = field(default=None, init=False, repr=False)

    def register_local_artifacts(self, location: Path) -> None:
        """Register a completed local artifact bundle."""
        self._local_artifacts = RunOutputLocator(location=location)

    def register_mlflow(self, run_id: str, location: str | None = None) -> None:
        """Register native MLflow output metadata for this execution."""
        self._mlflow = RunOutputLocator(location=location, native_run_id=run_id)

    @property
    def completed_run(self) -> CompletedRunSummary:
        """Return the currently known immutable output summary."""
        return CompletedRunSummary(
            self.identity,
            local_artifacts=self._local_artifacts,
            mlflow=self._mlflow,
        )
