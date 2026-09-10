import numpy as np
from pyts.image import GramianAngularField
from skimage.transform import resize

from detectiv.ts2i.transformations import GASF, GAFOutputNormalization


def test_gasf_matches_spiral_output_normalization() -> None:
    values = np.array([0.0, 0.2, 0.6, 1.0])

    image = GASF().transform(values, (5, 6))

    expected = GramianAngularField(method="summation").fit_transform(
        values[np.newaxis, :]
    )[0]
    expected = np.clip((expected + 1) / 2, 0, 1)
    expected = resize(  # type: ignore[no-untyped-call]
        expected,
        (5, 6),
        anti_aliasing=True,
        preserve_range=True,
    )
    np.testing.assert_array_equal(image, np.asarray(expected, dtype=np.float32))


def test_gasf_can_preserve_the_prism_output_range() -> None:
    values = np.array([0.0, 0.2, 0.6, 1.0])

    image = GASF(GAFOutputNormalization.NONE).transform(values, (4, 4))

    assert image.min() < 0
    assert image.max() <= 1
