from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from detectiv.callbacks.base import BaseCallback
from detectiv.scenarios.artifacts import RunArtifacts, RunArtifactWriter

if TYPE_CHECKING:
    from detectiv.scenarios.reconstruction import ReconstructionScenarioResult


class RunArtifactCallback(BaseCallback):
    def __init__(
        self,
        directory: Path,
        *,
        metrics_provider: Callable[[], Mapping[str, object]] | None = None,
        provenance: Mapping[str, object] | None = None,
        visualize: bool = False,
    ) -> None:
        self.writer = RunArtifactWriter(
            directory, provenance=provenance, visualize=visualize
        )
        self.metrics_provider = metrics_provider
        self.artifacts: RunArtifacts | None = None

    @property
    def name(self) -> str:
        return "artifacts"

    def on_run_started(self) -> None:
        self.artifacts = None

    def on_run_finished(self, result: "ReconstructionScenarioResult") -> None:
        self.artifacts = self.writer.write(
            result,
            metrics=None if self.metrics_provider is None else self.metrics_provider(),
        )
