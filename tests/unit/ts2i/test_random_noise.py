import numpy as np

from detectiv import Identity, ProjectionScheme, RandomNoise


def test_random_noise_is_rendered_as_an_image_plane() -> None:
    image = (
        ProjectionScheme(Identity())
        .channels(RandomNoise())
        .render(np.arange(8).reshape(4, 2), (5, 7))
    )

    assert image.shape == (1, 5, 7)
    assert image.dtype == np.float32
