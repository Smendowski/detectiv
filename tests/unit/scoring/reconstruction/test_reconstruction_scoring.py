import numpy as np
from torch import Tensor

from detectiv.images import ImageDataset, ImageShape, ImageSource
from detectiv.models.autoencoders import Autoencoder
from detectiv.models.autoencoders.base import ImageDecoder, ImageEncoder
from detectiv.scoring.reconstruction import MeanSquaredWindowReconstructionError
from detectiv.time_series.windowing import WindowReference


class ArrayImageSource(ImageSource):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> np.ndarray:
        return np.ones((1, 2, 4))


class IdentityEncoder(ImageEncoder):
    def forward(self, images: Tensor) -> Tensor:
        return images


class ZeroDecoder(ImageDecoder):
    def forward(self, embeddings: Tensor) -> Tensor:
        return embeddings * 0


def test_window_error_collapses_the_reconstruction_error() -> None:
    images = ImageDataset(
        "images",
        image_shape=ImageShape(1, 2, 4),
        window_references=(WindowReference("series", 0, 4, 4),),
        source=ArrayImageSource(),
        series_id="series",
        series_length=4,
    )

    scores = MeanSquaredWindowReconstructionError().score(
        Autoencoder(IdentityEncoder(), ZeroDecoder()), images
    )

    np.testing.assert_allclose(scores.values, [1.0])
