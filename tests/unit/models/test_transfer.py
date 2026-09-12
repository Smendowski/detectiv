import torch

from detectiv.models.autoencoders import (
    Autoencoder,
    DifferentialLearningRateStrategy,
    FrozenEncoderStrategy,
    ProgressiveEncoderUnfreezeStrategy,
)
from detectiv.models.autoencoders.decoders import CNNDecoder
from detectiv.models.autoencoders.encoders import CNNEncoder


def test_frozen_encoder_uses_the_supplied_pytorch_optimizer() -> None:
    model = _model()

    optimizer = FrozenEncoderStrategy().initialize(model, torch.optim.Adam, 1e-3, 0.0)

    assert isinstance(optimizer, torch.optim.Adam)
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(
        id(parameter) not in {id(item) for item in model.encoder.parameters()}
        for group in optimizer.param_groups
        for parameter in group["params"]
    )


def test_progressive_unfreeze_rebuilds_a_differential_optimizer() -> None:
    model = _model()
    strategy = ProgressiveEncoderUnfreezeStrategy(unfreeze_epoch=1)
    optimizer = strategy.initialize(model, torch.optim.Adam, 1e-3, 0.0)

    updated = strategy.on_epoch_started(
        1,
        model,
        optimizer,
        torch.optim.Adam,
        1e-3,
        0.0,
    )

    assert isinstance(updated, torch.optim.Adam)
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())
    assert [group["lr"] for group in updated.param_groups] == [1e-4, 1e-3]


def test_differential_learning_rate_strategy_unfreezes_the_encoder() -> None:
    model = _model()
    model.encoder.freeze()

    optimizer = DifferentialLearningRateStrategy(0.2).initialize(
        model,
        torch.optim.Adam,
        1e-3,
        0.0,
    )

    assert all(parameter.requires_grad for parameter in model.encoder.parameters())
    assert [group["lr"] for group in optimizer.param_groups] == [2e-4, 1e-3]


def _model() -> Autoencoder:
    torch.manual_seed(0)
    return Autoencoder(
        CNNEncoder(1, hidden_channels=(4,)),
        CNNDecoder(4, hidden_channels=(), output_channels=1),
    )
