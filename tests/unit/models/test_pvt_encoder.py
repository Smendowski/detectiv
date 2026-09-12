import torch

from detectiv.models.autoencoders.encoders import PVTv2B1Encoder


def test_pvtv2_b1_encoder_matches_the_published_feature_shape() -> None:
    encoder = PVTv2B1Encoder(pretrained=False, frozen=True)

    features = encoder(torch.rand(1, 3, 64, 64))

    assert features.shape == (1, 512, 2, 2)
    assert not any(parameter.requires_grad for parameter in encoder.parameters())


def test_pvtv2_b1_encoder_accepts_an_arbitrary_image_channel_count() -> None:
    encoder = PVTv2B1Encoder(input_channels=5, pretrained=False)

    features = encoder(torch.rand(1, 5, 64, 64))

    assert features.shape == (1, 512, 2, 2)
