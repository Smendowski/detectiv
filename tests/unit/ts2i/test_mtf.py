import numpy as np
from pyts.image import MarkovTransitionField
from skimage.transform import resize

from detectiv.ts2i.transformations import MTF


def test_mtf_matches_the_spiral_primitive() -> None:
    values = np.array([0.0, 0.2, 0.6, 1.0])

    image = MTF().transform(values, (5, 6))

    expected = MarkovTransitionField().fit_transform(values[np.newaxis, :])[0]
    expected = resize(  # type: ignore[no-untyped-call]
        expected,
        (5, 6),
        anti_aliasing=True,
        preserve_range=True,
    )
    expected = np.clip(np.asarray(expected, dtype=np.float32), 0, 1)
    np.testing.assert_array_equal(image, expected)


def test_mtf_accepts_a_single_feature_window() -> None:
    image = MTF().transform(np.array([[0.0], [0.5], [1.0]]), (4, 4))

    assert image.shape == (4, 4)
