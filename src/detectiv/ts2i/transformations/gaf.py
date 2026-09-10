from enum import StrEnum

import numpy as np
from pyts.image import GramianAngularField
from skimage.transform import resize

from detectiv.ts2i.transformations.base import (
    TransformationInput,
    TS2ITransformation,
    univariate_values,
)
from detectiv.ts2i.transformations.registry import register_transformation


class GAFOutputNormalization(StrEnum):
    NONE = "none"
    UNIT_INTERVAL = "unit_interval"


class _GAF(TS2ITransformation):
    method: str

    def __init__(
        self,
        output_normalization: GAFOutputNormalization = (
            GAFOutputNormalization.UNIT_INTERVAL
        ),
    ) -> None:
        self.output_normalization = output_normalization

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
        series = univariate_values(values, "GAF")
        image = GramianAngularField(method=self.method).fit_transform(
            series[np.newaxis, :]
        )[0]
        if self.output_normalization is GAFOutputNormalization.UNIT_INTERVAL:
            image = np.clip((image + 1) / 2, 0, 1)
        image = resize(  # type: ignore[no-untyped-call]
            image,
            size,
            anti_aliasing=True,
            preserve_range=True,
        )
        return np.asarray(image, dtype=np.float32)


@register_transformation("GASF")
class GASF(_GAF):
    method = "summation"


@register_transformation("GADF")
class GADF(_GAF):
    method = "difference"
