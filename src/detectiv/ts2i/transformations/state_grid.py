import numpy as np
from skimage.transform import resize

from detectiv.ts2i.transformations.base import (
    TransformationInput,
    TS2ITransformation,
)
from detectiv.ts2i.transformations.registry import register_transformation


@register_transformation("SG")
class StateGrid(TS2ITransformation):
    """Resize univariate or multivariate values directly into an image plane."""

    @property
    def input_kinds(self) -> frozenset[TransformationInput]:
        """Return the univariate and multivariate input requirements.

        Returns:
            Both input categories accepted by direct resizing.
        """
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
        """Resize source values directly into an image plane.

        Args:
            values: One-feature or multifeature time-series values.
            size: Output image height and width.
            rng: Unused random generator accepted by the common contract.

        Returns:
            A float32 image plane.
        """
        source = values[:, np.newaxis] if values.ndim == 1 else values
        image = resize(  # type: ignore[no-untyped-call]
            source,
            size,
            anti_aliasing=True,
            preserve_range=True,
        )
        return np.asarray(image, dtype=np.float32)
