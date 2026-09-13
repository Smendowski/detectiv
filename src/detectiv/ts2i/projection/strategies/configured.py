from copy import deepcopy
from dataclasses import replace

from detectiv.time_series import TimeSeriesDataset
from detectiv.ts2i.projection.schemes import ProjectionScheme
from detectiv.ts2i.projection.strategies.base import ProjectionStrategy


class ConfiguredProjectionStrategy(ProjectionStrategy):
    def __init__(self, scheme: ProjectionScheme) -> None:
        self.scheme = scheme

    def fit(self, train: TimeSeriesDataset) -> ProjectionScheme:
        channelization = deepcopy(self.scheme.channelization)
        channelization.fit(train)
        return replace(self.scheme, channelization=channelization)
