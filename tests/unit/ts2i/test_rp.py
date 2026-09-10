import numpy as np
from pyts.image import RecurrencePlot
from skimage.transform import resize

from detectiv.ts2i.transformations import RP


def test_rp_matches_spiral_and_prism_defaults() -> None:
    values = np.array([0.0, 0.2, 0.6, 1.0])

    image = RP().transform(values, (5, 6))

    expected = RecurrencePlot(threshold="point", percentage=10).fit_transform(
        values[np.newaxis, :]
    )[0]
    expected = resize(  # type: ignore[no-untyped-call]
        expected,
        (5, 6),
        anti_aliasing=True,
        preserve_range=True,
    )
    expected = np.clip(np.asarray(expected, dtype=np.float32), 0, 1)
    np.testing.assert_array_equal(image, expected)
