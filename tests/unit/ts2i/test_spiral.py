import numpy as np

from detectiv.ts2i.transformations import Spiral


def test_spiral_matches_the_published_mapping() -> None:
    values = np.array([0.0, 0.5, 1.0])

    image = Spiral().transform(values, (4, 4))

    center = 2
    y_coordinates, x_coordinates = np.ogrid[:4, :4]
    dx = x_coordinates - center
    dy = y_coordinates - center
    radius = np.sqrt(dx * dx + dy * dy)
    mask = radius <= 2
    theta = np.arctan2(dy, dx)
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)
    indices = (((theta + radius / 2 * 4 * np.pi) / (2 * np.pi)) * 3 / 2).astype(int)
    expected = np.zeros((4, 4), dtype=np.float32)
    expected[mask] = values[indices[mask] % 3]
    np.testing.assert_array_equal(image, expected)
