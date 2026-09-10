import numpy as np
from skimage.transform import resize

from detectiv.ts2i.transformations.base import (
    TransformationInput,
    TS2ITransformation,
)
from detectiv.ts2i.transformations.registry import register_transformation


@register_transformation("SG")
class StateGrid(TS2ITransformation):
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
        source = values[:, np.newaxis] if values.ndim == 1 else values
        image = resize(  # type: ignore[no-untyped-call]
            source,
            size,
            anti_aliasing=True,
            preserve_range=True,
        )
        return np.asarray(image, dtype=np.float32)
