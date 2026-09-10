import numpy as np
from pyts.image import MarkovTransitionField
from skimage.transform import resize

from detectiv.ts2i.transformations.base import (
    TransformationInput,
    TS2ITransformation,
    univariate_values,
)
from detectiv.ts2i.transformations.registry import register_transformation


@register_transformation("MTF")
class MTF(TS2ITransformation):
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
        series = univariate_values(values, "MTF")
        image = MarkovTransitionField().fit_transform(series[np.newaxis, :])[0]
        image = resize(  # type: ignore[no-untyped-call]
            image,
            size,
            anti_aliasing=True,
            preserve_range=True,
        )
        return np.clip(np.asarray(image, dtype=np.float32), 0, 1)
