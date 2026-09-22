import numpy as np
import pytest
import pywt
from skimage.transform import resize

from detectiv.images import ImageSize
from detectiv.ts2i.transformations import RWT, WaveletOutputNormalization


def test_rwt_matches_the_spiral_primitive() -> None:
    values = np.linspace(0, 1, 20)

    image = RWT().transform(values, ImageSize(height=5, width=6))

    coefficients, _ = pywt.cwt(values.astype(np.float32), np.arange(1, 17), "mexh")
    expected = np.log1p(np.abs(coefficients)).astype(np.float32)
    expected = resize(  # type: ignore[no-untyped-call]
        expected,
        (5, 6),
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)
    expected = (expected - expected.min()) / (expected.max() - expected.min() + 1e-8)
    np.testing.assert_allclose(image, np.clip(expected, 0, 1), rtol=1e-6, atol=1e-7)


def test_rwt_can_preserve_prism_wavelet_magnitudes() -> None:
    values = np.linspace(0, 1, 20)

    image = RWT(WaveletOutputNormalization.NONE).transform(
        values, ImageSize(height=5, width=6)
    )

    assert image.max() > 1


def test_rwt_can_clip_like_prism_image_storage() -> None:
    values = np.linspace(0, 1, 20)

    image = RWT(WaveletOutputNormalization.CLIP_UNIT_INTERVAL).transform(
        values, ImageSize(height=5, width=6)
    )

    assert image.min() >= 0
    assert image.max() <= 1


def test_rwt_normalizes_string_configuration_values() -> None:
    assert RWT("none").output_normalization is WaveletOutputNormalization.NONE

    with pytest.raises(ValueError, match="unsupported"):
        RWT("unsupported")
