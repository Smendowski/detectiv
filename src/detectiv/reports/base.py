from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import ClassVar, Self, cast

import numpy as np

from detectiv.typing import JSONValue


@dataclass(frozen=True, kw_only=True)
class ExperimentReport:
    """Common execution metadata shared by scenario-specific reports."""

    scenario_type: ClassVar[str]
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
