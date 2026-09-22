import numpy as np

from detectiv.images import ImageSize
from detectiv.ts2i.transformations import LineGrid


def test_line_grid_renders_one_patch_per_feature() -> None:
    values = np.array([[0.0, 1.0], [1.0, 0.0]])

    image = LineGrid().transform(values, ImageSize(height=8, width=8))

    assert image.shape == (8, 8)
    assert image.dtype == np.float32
    assert image.min() >= 0
    assert image.max() <= 1
