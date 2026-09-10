import numpy as np
from skimage.transform import resize

from detectiv.ts2i.transformations import StateGrid


def test_state_grid_matches_the_legacy_resize_semantics() -> None:
    values = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])

    image = StateGrid().transform(values, (4, 5))

    expected = resize(  # type: ignore[no-untyped-call]
        values,
        (4, 5),
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)
    np.testing.assert_array_equal(image, expected)


def test_state_grid_accepts_a_univariate_signal() -> None:
    image = StateGrid().transform(np.array([0.0, 1.0, 2.0]), (4, 5))

    assert image.shape == (4, 5)
