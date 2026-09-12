import numpy as np

from detectiv.ts2i.channelization import IdentityChannelization
from detectiv.ts2i.channelization.context import WindowContext
from detectiv.ts2i.projection import ProjectionScheme
from detectiv.ts2i.transformations import Spiral, SpiralInputNormalization


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


def test_spiral_can_normalize_each_input_window() -> None:
    image = Spiral(
        input_normalization=SpiralInputNormalization.UNIT_INTERVAL
    ).transform(np.array([2.0, 3.0, 4.0]), (4, 4))

    assert image.min() == 0
    assert image.max() == 1


def test_spiral_accepts_a_single_feature_window() -> None:
    image = (
        ProjectionScheme(IdentityChannelization(WindowContext()))
        .channels(Spiral())
        .replicate(n_channels=3)
        .render(np.arange(8, dtype=np.float32).reshape(8, 1), (8, 8))
    )

    assert image.shape == (3, 8, 8)
    np.testing.assert_array_equal(image[0], image[1])
