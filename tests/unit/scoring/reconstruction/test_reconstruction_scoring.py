import numpy as np
from torch import Tensor

from detectiv.data import WindowReference
from detectiv.datasets import ImageDataset, ImageShape, ImageSource
from detectiv.models.autoencoders import Autoencoder
from detectiv.models.autoencoders.base import ImageDecoder, ImageEncoder
from detectiv.scoring.reconstruction import (
    MeanSquaredTemporalColumnError,
    MeanSquaredWindowError,
)


class ArrayImageSource(ImageSource):
    def __init__(self, values: list[np.ndarray]) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.values[index]


class IdentityEncoder(ImageEncoder):
    def forward(self, images: Tensor) -> Tensor:
        return images


class ZeroDecoder(ImageDecoder):
    def forward(self, embeddings: Tensor) -> Tensor:
        return embeddings * 0


def test_window_error_collapses_the_reconstruction_error() -> None:
    images = _images()

    scores = MeanSquaredWindowError().score(_zero_autoencoder(), images)

    np.testing.assert_allclose(scores.values, [7.5])


def test_temporal_column_error_retains_time_resolved_error() -> None:
    images = _images()

    scores = MeanSquaredTemporalColumnError().score(_zero_autoencoder(), images)

    np.testing.assert_allclose(scores.values[0], [1.0, 4.0, 9.0, 16.0])


def _zero_autoencoder() -> Autoencoder:
    return Autoencoder(IdentityEncoder(), ZeroDecoder())


def _images() -> ImageDataset:
    return ImageDataset(
        "images",
        image_shape=ImageShape(1, 2, 4),
        window_references=(WindowReference("series", 0, 4, 4),),
        source=ArrayImageSource(
            [np.array([[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]])]
        ),
        series_lengths={"series": 4},
    )
