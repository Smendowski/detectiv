from enum import StrEnum

import numpy as np

from detectiv.ts2i.transformations.base import (
    TransformationInput,
    TS2ITransformation,
    univariate_values,
)
from detectiv.ts2i.transformations.registry import register_transformation


class SpiralInputNormalization(StrEnum):
    """Input scaling applied before placing values on a spiral."""

    NONE = "none"
    UNIT_INTERVAL = "unit_interval"


@register_transformation("SPIRAL")
class Spiral(TS2ITransformation):
    """Map a univariate series onto a radial spiral image."""

    def __init__(
        self,
        arms: int = 2,
        input_normalization: SpiralInputNormalization | str = (
            SpiralInputNormalization.NONE
        ),
    ) -> None:
        """Configure the number of spiral arms and optional input scaling."""
        if arms <= 0:
            raise ValueError("arms must be positive")
        self.arms = arms
        self.input_normalization = SpiralInputNormalization(input_normalization)

    @property
    def input_kinds(self) -> frozenset[TransformationInput]:
        """Return the univariate input requirement.

        Returns:
            The univariate input category.
        """
        return frozenset({TransformationInput.UNIVARIATE})

    def transform(
        self,
        values: np.ndarray,
        size: tuple[int, int],
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Map a univariate sequence to the configured radial layout.

        Args:
            values: One-feature time-series values.
            size: Output image height and width.
            rng: Unused random generator accepted by the common contract.

        Returns:
            A float32 image plane.
        """
        series = univariate_values(values, "Spiral")
        if self.input_normalization is SpiralInputNormalization.UNIT_INTERVAL:
            series = self._normalize(series)
        height, width = size
        radius_limit = min(height, width) / 2
        rows, columns = np.ogrid[:height, :width]
        distance = np.hypot(rows - height / 2, columns - width / 2)
        within_disc = distance <= radius_limit

        angle = np.arctan2(rows - height / 2, columns - width / 2)
        angle = np.where(angle < 0, angle + 2 * np.pi, angle)
        phase = angle + distance / radius_limit * 2 * np.pi * self.arms
        indices = ((phase / (2 * np.pi)) * series.size / self.arms).astype(np.intp)

        image = np.zeros((height, width), dtype=np.float32)
        image[within_disc] = series[indices[within_disc] % series.size]
        return image

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        minimum = values.min()
        maximum = values.max()
        if maximum == minimum:
            return np.zeros_like(values)
        return np.asarray((values - minimum) / (maximum - minimum), dtype=np.float32)
