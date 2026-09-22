from abc import ABC, abstractmethod

from detectiv.time_series import TimeSeries
from detectiv.ts2i.projection.schemes import ProjectionScheme


class ProjectionStrategy(ABC):
    """Fit training-dependent state and provide a usable projection scheme."""

    @abstractmethod
    def fit(self, train: TimeSeries) -> ProjectionScheme:
        """Fit on the training split and return its projection scheme.

        Args:
            train: Training-only data used to fit projection state.

        Returns:
            A scheme ready to render every dataset split.

        Raises:
            NotImplementedError: If a concrete strategy does not implement it.
        """
        raise NotImplementedError
