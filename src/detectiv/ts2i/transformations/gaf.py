from enum import StrEnum

import numpy as np
from pyts.image import GramianAngularField
from skimage.transform import resize

from detectiv.images import ImageSize
from detectiv.ts2i.transformations.base import (
    TransformationInput,
    TS2ITransformation,
    univariate_values,
)
from detectiv.ts2i.transformations.registry import register_transformation


class GAFOutputNormalization(StrEnum):
    """Output scaling applied to Gramian angular fields."""

    NONE = "none"
    UNIT_INTERVAL = "unit_interval"


class _GAF(TS2ITransformation):
    method: str

    def __init__(
        self,
        output_normalization: GAFOutputNormalization | str = (
            GAFOutputNormalization.UNIT_INTERVAL
        ),
    ) -> None:
        """Configure optional scaling of the rendered field.

        Args:
            output_normalization: Keep signed values or scale them to ``[0, 1]``.
        """
        self.output_normalization = GAFOutputNormalization(output_normalization)

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
        size: ImageSize,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Render a Gramian angular field at the requested size.

        Args:
            values: One-feature time-series values.
            size: Output image height and width.
            rng: Unused random generator accepted by the common contract.

        Returns:
            A float32 image plane.
        """
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
    """Render a univariate series as a Gramian angular summation field."""

    method = "summation"


@register_transformation("GADF")
class GADF(_GAF):
    """Render a univariate series as a Gramian angular difference field."""

    method = "difference"
