import numpy as np

from detectiv.datasets import TimeSeriesDataset
from detectiv.ts2i.projection.schemes import ProjectionScheme
from detectiv.ts2i.projection.strategies.base import ProjectionStrategy


class FixedProjectionStrategy(ProjectionStrategy):
    def __init__(self, scheme: ProjectionScheme) -> None:
        self.scheme = scheme

    @property
    def n_channels(self) -> int:
        return self.scheme.n_channels

    def fit(self, train: TimeSeriesDataset) -> "FixedProjectionStrategy":
        self.scheme.fit(train)
        return self

    def render(
        self,
        window: np.ndarray,
        size: tuple[int, int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        return self.scheme.render(window, size, rng=rng)
