import numpy as np
import pytest

from detectiv.images import ImageDataset, ImageShape, ImageSource
from detectiv.models.autoencoders import Autoencoder, AutoencoderTrainer
from detectiv.models.autoencoders.decoders import CNNDecoder
from detectiv.models.autoencoders.encoders import CNNEncoder
from detectiv.time_series.windowing import WindowReference


class ArrayImageSource(ImageSource):
    def __init__(self, images: list[np.ndarray]) -> None:
        self._images = images

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> np.ndarray:
        return self._images[index]


def test_trainer_rejects_empty_validation_selection() -> None:
    with pytest.raises(ValueError, match="at least one validation image"):
        AutoencoderTrainer(device="cpu").fit(
            _model(),
            _images(),
            [0],
            validation=_images(),
            validation_indices=(),
        )


def test_trainer_records_best_validation_epoch() -> None:
    history = AutoencoderTrainer(epochs=1, batch_size=1, device="cpu").fit(
        _model(),
        _images(),
        [0],
        validation=_images(),
        validation_indices=[0],
    )

    assert history.best_epoch == 0
    assert history.best_validation_loss == history.validation_losses[0]


def _images() -> ImageDataset:
    image = np.ones((1, 4, 4), dtype=np.float32)
    return ImageDataset(
        "images",
        image_shape=ImageShape(1, 4, 4),
        window_references=[WindowReference("series", 0, 4, 4)],
        source=ArrayImageSource([image]),
    )


def _model() -> Autoencoder:
    return Autoencoder(
        CNNEncoder(1, hidden_channels=(2,)),
        CNNDecoder(2, hidden_channels=(), output_channels=1),
    )
