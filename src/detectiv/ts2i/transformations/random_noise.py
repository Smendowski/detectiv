import numpy as np

from detectiv.ts2i.transformations.base import (
    TransformationInput,
    TS2ITransformation,
)


class RandomNoise(TS2ITransformation):
    @property
    def input_kinds(self) -> frozenset[TransformationInput]:
        return frozenset(
            {TransformationInput.UNIVARIATE, TransformationInput.MULTIVARIATE}
        )

    def transform(
        self,
        values: np.ndarray,
        size: tuple[int, int],
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        generator = np.random.default_rng() if rng is None else rng
        return generator.standard_normal(size, dtype=np.float32)
