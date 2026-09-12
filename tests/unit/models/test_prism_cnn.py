from typing import cast

import torch
from torch import nn

from detectiv.models.autoencoders import Autoencoder
from detectiv.models.autoencoders.decoders import CNNDecoder
from detectiv.models.autoencoders.encoders import CNNEncoder


def test_prism_cnn_architecture_is_representable() -> None:
    model = Autoencoder(
        CNNEncoder(
            input_channels=3,
            hidden_channels=(16, 32, 64),
            use_batch_norm=False,
        ),
        CNNDecoder(
            input_channels=64,
            hidden_channels=(32, 16),
            output_channels=3,
            use_batch_norm=False,
            output_activation=nn.Sigmoid,
        ),
    )

    reconstruction = model.reconstruct(torch.rand(2, 3, 64, 64))
    encoder_layers = tuple(cast(nn.Sequential, model.encoder.layers).children())
    decoder_layers = tuple(cast(nn.Sequential, model.decoder.layers).children())

    assert reconstruction.shape == (2, 3, 64, 64)
    assert isinstance(encoder_layers[0], nn.Conv2d)
    assert isinstance(encoder_layers[1], nn.ReLU)
    assert isinstance(decoder_layers[0], nn.ConvTranspose2d)
    assert isinstance(decoder_layers[-1], nn.Sigmoid)
    assert isinstance(encoder_layers[0], nn.Conv2d)
    assert isinstance(encoder_layers[2], nn.Conv2d)
    assert isinstance(encoder_layers[4], nn.Conv2d)
    assert encoder_layers[0].weight.shape == (16, 3, 3, 3)
    assert encoder_layers[2].weight.shape == (32, 16, 3, 3)
    assert encoder_layers[4].weight.shape == (64, 32, 3, 3)
