import numpy as np

from detectiv.ts2i.transformations.base import (
    TransformationInput,
    TS2ITransformation,
    univariate_values,
)
from detectiv.ts2i.transformations.registry import register_transformation


@register_transformation("SPIRAL")
class Spiral(TS2ITransformation):
    def __init__(self, arms: int = 2) -> None:
        if arms <= 0:
            raise ValueError("arms must be positive")
        self.arms = arms

    @property
    def input_kinds(self) -> frozenset[TransformationInput]:
        return frozenset({TransformationInput.UNIVARIATE})

    def transform(
        self,
        values: np.ndarray,
        size: tuple[int, int],
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        series = univariate_values(values, "Spiral")
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
