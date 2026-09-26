from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from detectiv.time_series import TimeSeries
from detectiv.time_series.windowing import WindowedTimeSeriesSplit
from detectiv.ts2i.projection.schemes import BaseProjectionScheme

if TYPE_CHECKING:
    from detectiv.ts2i.preparation import ProjectedImageStage


class BaseProjectionStrategy(ABC):
    """Fit training-dependent state and provide a usable projection scheme."""

    def project(self, source: WindowedTimeSeriesSplit) -> ProjectedImageStage:
        """Create the projected execution stage for windowed partitions.

        Args:
            source: Windowed temporal partitions to project.

        Returns:
            Projected image stage configured with this strategy.
        """
        from detectiv.ts2i.preparation import ProjectedImageStage

        return ProjectedImageStage(source=source, projection=self)

    @abstractmethod
    def fit(self, train: TimeSeries) -> BaseProjectionScheme:
        """Fit on the training split and return its projection scheme.

        Args:
            train: Training-only data used to fit projection state.

        Returns:
            A scheme ready to render every dataset split.

        Raises:
            NotImplementedError: If a concrete strategy does not implement it.
        """
        raise NotImplementedError
