import numpy as np
from pyts.image import MarkovTransitionField
from skimage.transform import resize

from detectiv.images import ImageSize
from detectiv.ts2i.transformations.base import (
    TransformationInput,
    TS2ITransformation,
    univariate_values,
)
from detectiv.ts2i.transformations.registry import register_transformation


@register_transformation("MTF")
class MTF(TS2ITransformation):
    """Render a univariate series as a Markov transition field."""

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
        """Render a Markov transition field at the requested size.

        Args:
            values: One-feature time-series values.
            size: Output image height and width.
            rng: Unused random generator accepted by the common contract.

        Returns:
            A float32 image plane normalized to ``[0, 1]``.
        """
        series = univariate_values(values, "MTF")
        image = MarkovTransitionField().fit_transform(series[np.newaxis, :])[0]
        image = resize(  # type: ignore[no-untyped-call]
            image,
            size,
            anti_aliasing=True,
            preserve_range=True,
        )
        return np.clip(np.asarray(image, dtype=np.float32), 0, 1)
