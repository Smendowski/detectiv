from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from detectiv.time_series.series import TimeSeriesSplit
from detectiv.time_series.windowing.core import WindowSpec


class SplitPart(StrEnum):
    """Named temporal partitions with independent window specifications."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class SplitWindowing:
    """Window specifications for train, optional validation, and test partitions."""

    def __init__(
        self,
        *,
        train: WindowSpec,
        test: WindowSpec,
        validation: WindowSpec | None = None,
    ) -> None:
        """Create split-specific windowing configuration."""
        self.train = train
        self.validation = validation
        self.test = test

    def spec_for(self, part: SplitPart | str) -> WindowSpec:
        """Return the specification configured for one split partition.

        Raises:
            ValueError: If the part is unknown or validation is not configured.
        """
        try:
            part = SplitPart(part)
        except ValueError as error:
            raise ValueError(f"unsupported split part: {part!r}") from error
        if part is SplitPart.TRAIN:
            return self.train
        if part is SplitPart.VALIDATION:
            if self.validation is None:
                raise ValueError("validation window specification is not configured")
            return self.validation
        if part is SplitPart.TEST:
            return self.test
        raise AssertionError("all split parts are handled")


class WindowProjection[T](Protocol):
    """Transition a windowed time-series split to a projection-specific result."""

    def project(self, source: WindowedTimeSeriesSplit) -> T:
        """Project ``source`` into a result owned by the implementing layer.

        Args:
            source: Windowed temporal partitions to project.

        Returns:
            Projection-specific result owned by the implementing layer.
        """
        ...


@dataclass(frozen=True)
class WindowedTimeSeriesSplit:
    """Temporal series split with configured per-partition windows."""

    split: TimeSeriesSplit
    windowing: SplitWindowing

    def project[T](self, projection: WindowProjection[T]) -> T:
        """Delegate this windowed split to a projection implementation.

        Args:
            projection: Projection implementation that owns the resulting stage.

        Returns:
            Result created by ``projection``.
        """
        return projection.project(self)
