from abc import ABC, abstractmethod

from detectiv.time_series import TimeSeriesDataset
from detectiv.ts2i.projection.schemes import ProjectionScheme


class ProjectionStrategy(ABC):
    @abstractmethod
    def fit(self, train: TimeSeriesDataset) -> ProjectionScheme:
        raise NotImplementedError
