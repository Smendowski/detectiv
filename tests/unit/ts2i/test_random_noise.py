import numpy as np

from detectiv.ts2i.channelization import IdentityChannelization
from detectiv.ts2i.channelization.context import WindowContext
from detectiv.ts2i.projection import ProjectionScheme
from detectiv.ts2i.transformations import RandomNoise


def test_random_noise_is_rendered_as_an_image_plane() -> None:
    image = (
        ProjectionScheme(IdentityChannelization(WindowContext()))
        .channels(RandomNoise())
        .render(np.arange(8).reshape(4, 2), (5, 7))
    )

    assert image.shape == (1, 5, 7)
    assert image.dtype == np.float32
