import numpy as np
from skimage.draw import disk, line
from skimage.transform import resize

from detectiv.ts2i.transformations.base import (
    TransformationInput,
    TS2ITransformation,
    univariate_values,
)
from detectiv.ts2i.transformations.registry import register_transformation


def _draw_line(
    values: np.ndarray,
    size: int,
    *,
    blank_single_value: bool = False,
) -> np.ndarray:
    image = np.zeros((size, size), dtype=np.float32)
    if values.size == 0:
        raise ValueError("line plotting requires at least one value")
    if values.size == 1 and blank_single_value:
        return image

    original_time = np.linspace(0, 1, values.size)
    target_time = np.linspace(0, 1, size)
    resampled = np.interp(target_time, original_time, values).astype(np.float32)
    rows = np.clip(
        np.rint((1 - np.clip(resampled, 0, 1)) * (size - 1)), 0, size - 1
    ).astype(np.intp)
    columns = np.arange(size, dtype=np.intp)

    for index in range(size - 1):
        row_indices, column_indices = line(  # type: ignore[no-untyped-call]
            rows[index], columns[index], rows[index + 1], columns[index + 1]
        )
        image[row_indices, column_indices] = 1
    for row, column in zip(rows, columns, strict=True):
        row_indices, column_indices = disk(  # type: ignore[no-untyped-call]
            (row, column), radius=1, shape=image.shape
        )
        image[row_indices, column_indices] = 1
    return image


@register_transformation("LP")
class LinePlot(TS2ITransformation):
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
        series = univariate_values(values, "LinePlot")
        source = _draw_line(series, max(size))
        image = resize(  # type: ignore[no-untyped-call]
            source,
            size,
            anti_aliasing=True,
            preserve_range=True,
        )
        return np.asarray(image, dtype=np.float32)
