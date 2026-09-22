import numpy as np

from detectiv.images import ImageSize
from detectiv.ts2i.transformations.base import (
    TransformationInput,
    TS2ITransformation,
)
from detectiv.ts2i.transformations.registry import register_transformation


@register_transformation("RN")
class RandomNoise(TS2ITransformation):
    """Render seeded Gaussian noise independent of the input values."""

    @property
    def input_kinds(self) -> frozenset[TransformationInput]:
        """Return the accepted univariate and multivariate input categories.

        Returns:
            Both input categories.
        """
        return frozenset(
            {TransformationInput.UNIVARIATE, TransformationInput.MULTIVARIATE}
        )

    def transform(
        self,
        values: np.ndarray,
        size: ImageSize,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate a normally distributed image independent of ``values``.

        Args:
            values: Ignored source values retained for contract compatibility.
            size: Output image height and width.
            rng: Optional generator controlling reproducible noise.

        Returns:
            A float32 Gaussian-noise image plane.
        """
        generator = np.random.default_rng() if rng is None else rng
        return generator.standard_normal(size, dtype=np.float32)
