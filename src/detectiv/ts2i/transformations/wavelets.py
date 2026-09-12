from enum import StrEnum

import numpy as np
import pywt
from skimage.transform import resize

from detectiv.ts2i.transformations.base import (
    TransformationInput,
    TS2ITransformation,
    univariate_values,
)
from detectiv.ts2i.transformations.registry import register_transformation


class WaveletOutputNormalization(StrEnum):
    NONE = "none"
    CLIP_UNIT_INTERVAL = "clip_unit_interval"
    UNIT_INTERVAL = "unit_interval"


class _ContinuousWavelet(TS2ITransformation):
    wavelet: str

    def __init__(
        self,
        output_normalization: WaveletOutputNormalization = (
            WaveletOutputNormalization.UNIT_INTERVAL
        ),
    ) -> None:
        self.output_normalization = output_normalization

    @property
    def input_kinds(self) -> frozenset[TransformationInput]:
        return frozenset({TransformationInput.UNIVARIATE})

    def transform(
        self,
        values: np.ndarray,
        size: tuple[int, int],
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
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
    wavelet = "mexh"


@register_transformation("MWT")
class MWT(_ContinuousWavelet):
    wavelet = "morl"
