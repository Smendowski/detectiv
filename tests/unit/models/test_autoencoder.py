import pytest
import torch

from detectiv.models.autoencoders import Autoencoder
from detectiv.models.autoencoders.bottlenecks import DenseSpatialBottleneck
from detectiv.models.autoencoders.decoders import CNNDecoder
from detectiv.models.autoencoders.encoders import CNNEncoder


def test_autoencoder_reconstructs_the_original_image_shape() -> None:
    encoder = CNNEncoder(3, hidden_channels=(8, 16))
    decoder = CNNDecoder(16, hidden_channels=(8,), output_channels=3)
    model = Autoencoder(encoder, decoder)

    reconstruction = model.reconstruct(torch.rand(2, 3, 17, 19))

    assert reconstruction.shape == (2, 3, 17, 19)


def test_cnn_encoder_accepts_a_configurable_number_of_layers() -> None:
    encoder = CNNEncoder(
        1,
        hidden_channels=(4, 8, 16),
        strides=(1, 2, 2),
        activation=torch.nn.GELU,
    )

    embeddings = encoder(torch.rand(2, 1, 16, 16))

    assert embeddings.shape == (2, 16, 4, 4)


def test_autoencoder_rejects_non_image_tensors() -> None:
    model = Autoencoder(
        CNNEncoder(1, hidden_channels=(4,)),
        CNNDecoder(4, hidden_channels=(), output_channels=1),
    )

    with pytest.raises(ValueError, match="batch, channels, height, width"):
        model.reconstruct(torch.rand(1, 16))


def test_autoencoder_accepts_a_dense_spatial_bottleneck() -> None:
    model = Autoencoder(
        CNNEncoder(1, hidden_channels=(4,)),
        CNNDecoder(4, hidden_channels=(), output_channels=1),
        bottleneck=DenseSpatialBottleneck(4, (4, 4), latent_dim=3),
    )

    reconstruction = model.reconstruct(torch.rand(2, 1, 8, 8))

    assert reconstruction.shape == (2, 1, 8, 8)
