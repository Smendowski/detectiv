import torch

from detectiv.models.autoencoders.encoders import ResNet18Encoder


def test_resnet18_encoder_matches_the_published_feature_shape() -> None:
    encoder = ResNet18Encoder(pretrained=False, frozen=True)

    features = encoder(torch.rand(2, 3, 64, 64))

    assert features.shape == (2, 512, 2, 2)
    assert not any(parameter.requires_grad for parameter in encoder.parameters())


def test_resnet18_encoder_can_be_unfrozen_for_fine_tuning() -> None:
    encoder = ResNet18Encoder(pretrained=False)

    encoder.unfreeze()

    assert all(parameter.requires_grad for parameter in encoder.parameters())


def test_resnet18_encoder_accepts_an_arbitrary_image_channel_count() -> None:
    encoder = ResNet18Encoder(input_channels=5, pretrained=False)

    features = encoder(torch.rand(2, 5, 64, 64))

    assert features.shape == (2, 512, 2, 2)
