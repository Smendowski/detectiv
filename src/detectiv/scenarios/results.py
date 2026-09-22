from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Self, cast

import numpy as np

if TYPE_CHECKING:
    from detectiv.callbacks import Callback
    from detectiv.models.autoencoders import TrainingHistory
    from detectiv.runs import JSONValue, RunArtifacts
    from detectiv.scoring import (
        PointScoringPlan,
        ReconstructionScoringPlan,
        WindowEvidenceBatch,
    )


@dataclass(frozen=True, kw_only=True)
class ExperimentReport[T]:
    """Common execution metadata shared by scenario-specific reports."""

    scenario_type: ClassVar[str]
    callbacks: Mapping[str, Callback[T]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    reproducibility: Mapping[str, JSONValue] = field(
        default_factory=lambda: MappingProxyType({})
    )
    resolved_inputs: Mapping[str, JSONValue] = field(
        default_factory=lambda: MappingProxyType({})
    )
    metrics: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        metrics: dict[str, float] = {}
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("metric names must not be empty")
            if isinstance(value, bool) or not isinstance(
                value, int | float | np.integer | np.floating
            ):
                raise ValueError(f"metric {name!r} must be numeric")
            metric = float(value)
            if not math.isfinite(metric):
                raise ValueError(f"metric {name!r} must be finite")
            metrics[name] = metric
        object.__setattr__(self, "metrics", MappingProxyType(metrics))

    def with_metrics(self, metrics: Mapping[str, float]) -> Self:
        """Return an immutable copy containing validated metric values.

        Args:
            metrics: Flat metric names mapped to finite numeric values.

        Returns:
            A copy of this report containing the supplied metrics.
        """
        return replace(self, metrics=metrics)

    @property
    def inputs(self) -> Mapping[str, JSONValue]:
        """Return recorded scenario inputs and their provenance.

        Returns:
            JSON-compatible resolved input configuration and provenance.
        """
        return self.resolved_inputs

    @property
    def performance(self) -> Mapping[str, JSONValue]:
        """Return recorded performance measurements, when available.

        Returns:
            JSON-compatible timing and resource measurements.
        """
        value = self.resolved_inputs.get("performance", {})
        return value if isinstance(value, Mapping) else MappingProxyType({})

    def record(self) -> dict[str, JSONValue]:
        """Return common JSON-serializable report metadata.

        Returns:
            Scenario type, resolved inputs, reproducibility, and optional metrics.
        """
        record: dict[str, JSONValue] = {
            "scenario_type": self.scenario_type,
            "resolved_inputs": self.resolved_inputs,
            "reproducibility": self.reproducibility,
        }
        if self.metrics:
            record["metrics"] = cast("JSONValue", self.metrics)
        return record

    def summary(self) -> str:
        """Return a concise human-readable report summary.

        Returns:
            A multiline summary suitable for terminal output.
        """
        lines = [f"Scenario: {self.scenario_type}"]
        lines.extend(self._metric_summary())
        return "\n".join(lines)

    def _metric_summary(self) -> list[str]:
        if not self.metrics:
            return []

        names = tuple(self.metrics)
        components = tuple(name.split(".") for name in names)
        prefix_length = 0
        for parts in zip(*components, strict=False):
            if len(set(parts)) != 1 or prefix_length >= min(
                len(name) - 1 for name in components
            ):
                break
            prefix_length += 1

        heading = "Metrics:"
        if prefix_length:
            prefix = " / ".join(components[0][:prefix_length])
            heading = f"Metrics ({prefix}):"
        lines = [heading]
        for name, value in self.metrics.items():
            short_name = ".".join(name.split(".")[prefix_length:])
            lines.append(f"  {short_name}: {value:.3f}")
        return lines


@dataclass(frozen=True)
class ReconstructionReport(ExperimentReport["ReconstructionReport"]):
    """Complete reconstruction outputs, configuration, and execution metadata."""

    scenario_type: ClassVar[str] = "reconstruction"

    window_scores: Mapping[str, WindowEvidenceBatch]
    point_scores: Mapping[str, Mapping[str, np.ndarray]]
    training: TrainingHistory
    point_labels: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.point_labels is None:
            return
        labels = np.array(self.point_labels, copy=True)
        labels.setflags(write=False)
        object.__setattr__(self, "point_labels", labels)

    @property
    def training_losses(self) -> tuple[float, ...]:
        """Return the recorded training loss for every completed epoch.

        Returns:
            Training losses in epoch order.
        """
        return self.training.training_losses

    @property
    def validation_losses(self) -> tuple[float, ...]:
        """Return the recorded validation loss for every completed epoch.

        Returns:
            Validation losses in epoch order.
        """
        return self.training.validation_losses

    def point_scores_for(
        self,
        *,
        scoring: ReconstructionScoringPlan | None = None,
        point_scoring: PointScoringPlan | None = None,
    ) -> np.ndarray:
        """Return point scores for optional configured plans.

        Args:
            scoring: Reconstruction plan to select when the result contains more
                than one.
            point_scoring: Propagation policy to select when the reconstruction
                plan contains more than one.

        Returns:
            Immutable point-level anomaly scores.

        Raises:
            ValueError: If either plan is ambiguous or the propagation policy does
                not belong to ``scoring``.
            KeyError: If the selected plan is absent from this result.
        """
        if scoring is None:
            if len(self.point_scores) != 1:
                raise ValueError(
                    "scoring is required when a result contains multiple "
                    "reconstruction scoring plans"
                )
            scoring_name = next(iter(self.point_scores))
        else:
            scoring_name = scoring.name

        propagated = self.point_scores[scoring_name]
        if point_scoring is None:
            if len(propagated) != 1:
                raise ValueError(
                    "point_scoring is required when a scoring plan contains "
                    "multiple point-scoring plans"
                )
            point_scoring_name = next(iter(propagated))
        elif scoring is not None and point_scoring not in scoring.point_scoring:
            raise ValueError("point_scoring must belong to scoring")
        else:
            point_scoring_name = point_scoring.name
        return propagated[point_scoring_name]

    def record(self) -> dict[str, JSONValue]:
        """Return JSON-serializable reconstruction report metadata.

        Returns:
            Common metadata plus reconstruction training history.
        """
        return {
            **super().record(),
            "training": {
                "losses": self.training_losses,
                "validation_losses": self.validation_losses,
                "best_epoch": self.training.best_epoch,
                "best_validation_loss": self.training.best_validation_loss,
                "device": self.training.device,
            },
        }

    def summary(self) -> str:
        """Return key input, training, scoring, and performance information.

        Returns:
            A multiline reconstruction report summary.
        """
        data = self.resolved_inputs.get("data")
        image_lines: list[str] = []
        if isinstance(data, Mapping):
            for split_name, label in (
                ("train", "Train"),
                ("validation", "Validation"),
                ("test", "Test"),
            ):
                split = data.get(split_name)
                count = (
                    split.get("window_count") if isinstance(split, Mapping) else None
                )
                if (
                    isinstance(count, int)
                    and not isinstance(count, bool)
                    and count >= 0
                ):
                    image_lines.append(f"{label} images: {count}")
                elif split_name == "validation" and split_name in data:
                    image_lines.append("Validation images: none")
        lines = ["Scenario: reconstruction"]
        provenance = self.resolved_inputs.get("input_provenance")
        if isinstance(provenance, Mapping):
            source = provenance.get("source")
            location = provenance.get("location")
            if source is not None:
                lines.append(f"Image source: {source}")
            if location is not None:
                lines.append(f"Image location: {location}")
        lines.extend(image_lines)
        lines.extend(
            (
                f"Training epochs: {len(self.training_losses)}",
                f"Device: {self.training.device or 'unknown'}",
                f"Scoring plans: {', '.join(self.point_scores) or 'none'}",
            )
        )
        lines.extend(self._metric_summary())
        return "\n".join(lines)

    def write(self, directory: Path, *, overwrite: bool = False) -> RunArtifacts:
        """Persist this report and its point scores as a verified run artifact.

        Args:
            directory: Destination directory for report artifacts.
            overwrite: Replace an existing non-empty destination.

        Returns:
            Paths to the report manifest and score artifacts.
        """
        from detectiv.runs import RunArtifactWriter

        return RunArtifactWriter(directory, overwrite=overwrite).write(self)


ReconstructionScenarioResult = ReconstructionReport
