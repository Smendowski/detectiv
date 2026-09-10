import numpy as np
from skimage.draw import disk, line

from detectiv.ts2i.transformations import LinePlot


def test_line_plot_matches_the_spiral_primitive() -> None:
    values = np.array([0.0, 0.4, 1.0])

    image = LinePlot().transform(values, (5, 5))

    expected = np.zeros((5, 5), dtype=np.float32)
    resampled = np.interp(np.linspace(0, 1, 5), np.linspace(0, 1, 3), values)
    rows = np.rint((1 - resampled) * 4).astype(int)
    for index in range(4):
        row_indices, column_indices = line(  # type: ignore[no-untyped-call]
            rows[index], index, rows[index + 1], index + 1
        )
        expected[row_indices, column_indices] = 1
    for row, column in enumerate(rows):
        row_indices, column_indices = disk(  # type: ignore[no-untyped-call]
            (row, column), radius=1, shape=expected.shape
        )
        expected[row_indices, column_indices] = 1
    np.testing.assert_array_equal(image, expected)


def test_line_plot_draws_a_single_observation() -> None:
    image = LinePlot().transform(np.array([0.5]), (5, 5))

    assert image.sum() > 0
