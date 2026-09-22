import numpy as np
from skimage.transform import resize

from detectiv.images import ImageSize
from detectiv.ts2i.transformations.base import (
    TransformationInput,
    TS2ITransformation,
)
from detectiv.ts2i.transformations.line_plot import _draw_line
from detectiv.ts2i.transformations.registry import register_transformation


@register_transformation("LG")
class LineGrid(TS2ITransformation):
    """Render each multivariate feature as a tiled line-plot cell."""

    @property
    def input_kinds(self) -> frozenset[TransformationInput]:
        """Return the multivariate input requirement.

        Returns:
            The multivariate input category.
        """
        return frozenset({TransformationInput.MULTIVARIATE})

    def transform(
        self,
        values: np.ndarray,
        size: ImageSize,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Tile one line plot per feature and resize the grid.

        Args:
            values: Two-dimensional time-major feature values.
            size: Output image height and width.
            rng: Unused random generator accepted by the common contract.

        Returns:
            A float32 image plane.

        Raises:
            ValueError: If no feature dimension is supplied.
        """
        if values.ndim != 2:
            raise ValueError("LineGrid requires two-dimensional values")
        n_features = values.shape[1]
        if n_features == 0:
            raise ValueError("LineGrid requires at least one feature")

        side = int(np.ceil(np.sqrt(n_features)))
        patch_size = max(2, max(size) // side)
        canvas = np.zeros((side * patch_size, side * patch_size), dtype=np.float32)
        for index in range(n_features):
            row, column = divmod(index, side)
            canvas[
                row * patch_size : (row + 1) * patch_size,
                column * patch_size : (column + 1) * patch_size,
            ] = _draw_line(values[:, index], patch_size, blank_single_value=True)
        image = resize(  # type: ignore[no-untyped-call]
            canvas,
            size,
            anti_aliasing=True,
            preserve_range=True,
        )
        return np.asarray(image, dtype=np.float32)
