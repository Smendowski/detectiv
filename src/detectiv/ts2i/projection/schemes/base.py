from dataclasses import dataclass, replace

import numpy as np

from detectiv.datasets import TimeSeriesDataset
from detectiv.ts2i.channelization import Channelization
from detectiv.ts2i.transformations import TransformationInput, TS2ITransformation


@dataclass(frozen=True)
class ProjectionScheme:
    channelization: Channelization
    _transformations: tuple[TS2ITransformation, ...] = ()
    _replication_count: int | None = None

    def channels(self, *transformations: TS2ITransformation) -> "ProjectionScheme":
        if not transformations:
            raise ValueError("at least one transformation is required")
        return replace(
            self,
            _transformations=transformations,
            _replication_count=None,
        )

    def replicate(self, *, n_channels: int) -> "ProjectionScheme":
        if n_channels <= 0:
            raise ValueError("n_channels must be positive")
        if len(self._transformations) != 1:
            raise ValueError("replicate requires exactly one transformation")
        return replace(self, _replication_count=n_channels)

    @property
    def n_channels(self) -> int:
        if self._replication_count is not None:
            return self._replication_count
        return len(self._transformations)

    def fit(self, train: TimeSeriesDataset) -> "ProjectionScheme":
        self.channelization.fit(train)
        return self

    def render(
        self,
        window: np.ndarray,
        size: tuple[int, int],
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if not self._transformations:
            raise RuntimeError("projection scheme must define transformations")
        inputs = self.channelization.transform(window)
        if self._replication_count is not None:
            if len(inputs) != 1:
                raise ValueError("replication requires one channelization output")
            plane = self._render_plane(inputs[0], self._transformations[0], size, rng)
            return np.stack((plane,) * self._replication_count)
        if len(inputs) != len(self._transformations):
            raise ValueError(
                f"channelization produced {len(inputs)} outputs; "
                f"{len(self._transformations)} transformations were configured"
            )
        return np.stack(
            [
                self._render_plane(values, transformation, size, rng)
                for values, transformation in zip(
                    inputs, self._transformations, strict=True
                )
            ]
        )

    @staticmethod
    def _render_plane(
        values: np.ndarray,
        transformation: TS2ITransformation,
        size: tuple[int, int],
        rng: np.random.Generator | None,
    ) -> np.ndarray:
        supports_univariate = (
            values.ndim == 1 or (values.ndim == 2 and values.shape[1] == 1)
        ) and TransformationInput.UNIVARIATE in transformation.input_kinds
        supports_multivariate = (
            values.ndim == 2
            and TransformationInput.MULTIVARIATE in transformation.input_kinds
        )
        if not (supports_univariate or supports_multivariate):
            raise ValueError("transformation does not support this feature shape")

        plane = transformation.transform(values, size, rng=rng)
        if plane.shape != size:
            raise ValueError(
                f"transformation returned {plane.shape}; expected image plane {size}"
            )
        return plane
