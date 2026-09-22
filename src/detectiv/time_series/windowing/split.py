from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from detectiv.time_series.series import TimeSeriesSplit
from detectiv.time_series.windowing.core import WindowSpec

if TYPE_CHECKING:
    from detectiv.ts2i.preparation import ProjectedImageStage
    from detectiv.ts2i.projection import ProjectionStrategy


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


@dataclass(frozen=True)
class WindowedTimeSeriesSplit:
    """Temporal series split with configured per-partition windows."""

    split: TimeSeriesSplit
    windowing: SplitWindowing

    def project(self, projection: ProjectionStrategy) -> ProjectedImageStage:
        """Transition to a projected image stage.

        Args:
            projection: Strategy to fit using the training partition.

        Returns:
            Projected stage that can build, inspect, or materialize images.
        """
        from detectiv.ts2i.preparation import ProjectedImageStage

        return ProjectedImageStage(source=self, projection=projection)
