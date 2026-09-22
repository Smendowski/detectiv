from enum import StrEnum

import numpy as np
import pywt
from skimage.transform import resize

from detectiv.images import ImageSize
from detectiv.ts2i.transformations.base import (
    TransformationInput,
    TS2ITransformation,
    univariate_values,
)
from detectiv.ts2i.transformations.registry import register_transformation


class WaveletOutputNormalization(StrEnum):
    """Output scaling applied to continuous wavelet scalograms."""

    NONE = "none"
    CLIP_UNIT_INTERVAL = "clip_unit_interval"
    UNIT_INTERVAL = "unit_interval"


class _ContinuousWavelet(TS2ITransformation):
    wavelet: str

    def __init__(
        self,
        output_normalization: WaveletOutputNormalization | str = (
            WaveletOutputNormalization.UNIT_INTERVAL
        ),
    ) -> None:
        """Configure optional scaling of the rendered scalogram.

        Args:
            output_normalization: Scaling applied after the wavelet transform.
        """
        self.output_normalization = WaveletOutputNormalization(output_normalization)

    @property
    def input_kinds(self) -> frozenset[TransformationInput]:
        """Return the univariate input requirement.

        Returns:
            The univariate input category.
        """
        return frozenset({TransformationInput.UNIVARIATE})

    def transform(
        self,
        values: np.ndarray,
        size: ImageSize,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Render a continuous wavelet scalogram at the requested size.

        Args:
            values: One-feature time-series values.
            size: Output image height and width.
            rng: Unused random generator accepted by the common contract.

        Returns:
            A float32 image plane.
        """
        series = univariate_values(values, "wavelet transforms")
        n_scales = max(2, min(max(16, series.size // 2), 64, series.size))
        scales = np.arange(1, n_scales + 1, dtype=np.float32)
        coefficients, _ = pywt.cwt(
            series.astype(np.float32),
            scales,
            wavelet=self.wavelet,
            sampling_period=1.0,
        )
        image = np.log1p(np.abs(np.asarray(coefficients, dtype=np.float32)))
        image = resize(  # type: ignore[no-untyped-call]
            image,
            size,
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)
        if self.output_normalization is WaveletOutputNormalization.CLIP_UNIT_INTERVAL:
            image = np.clip(image, 0, 1)
        elif self.output_normalization is WaveletOutputNormalization.UNIT_INTERVAL:
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            image = np.clip(image, 0, 1)
        return np.asarray(image, dtype=np.float32)


@register_transformation("RWT")
class RWT(_ContinuousWavelet):
    """Render a Ricker-wavelet scalogram from a univariate series."""

    wavelet = "mexh"


@register_transformation("MWT")
class MWT(_ContinuousWavelet):
    """Render a Morlet-wavelet scalogram from a univariate series."""

    wavelet = "morl"
