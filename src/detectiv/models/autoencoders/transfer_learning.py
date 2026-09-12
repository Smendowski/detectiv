from collections.abc import Callable, Iterator

from torch import nn, optim

from detectiv.models.autoencoders.model import Autoencoder

OptimizerFactory = Callable[..., optim.Optimizer]


class TransferLearningStrategy:
    def initialize(
        self,
        model: Autoencoder,
        optimizer_type: OptimizerFactory,
        learning_rate: float,
        weight_decay: float,
    ) -> optim.Optimizer:
        return _optimizer(
            optimizer_type,
            model.parameters(),
            learning_rate,
            weight_decay,
        )

    def on_epoch_started(
        self,
        epoch: int,
        model: Autoencoder,
        optimizer: optim.Optimizer,
        optimizer_type: OptimizerFactory,
        learning_rate: float,
        weight_decay: float,
    ) -> optim.Optimizer:
        return optimizer


class FrozenEncoderStrategy(TransferLearningStrategy):
    def initialize(
        self,
        model: Autoencoder,
        optimizer: OptimizerFactory,
        learning_rate: float,
        weight_decay: float,
    ) -> optim.Optimizer:
        model.encoder.freeze()
        return _optimizer(
            optimizer,
            _non_encoder_parameters(model),
            learning_rate,
            weight_decay,
        )


class ProgressiveEncoderUnfreezeStrategy(FrozenEncoderStrategy):
    def __init__(
        self,
        unfreeze_epoch: int = 20,
        encoder_learning_rate_scale: float = 0.1,
    ) -> None:
        if unfreeze_epoch < 0 or not 0 < encoder_learning_rate_scale <= 1:
            raise ValueError(
                "unfreeze_epoch must be non-negative and encoder learning-rate "
                "scale in (0, 1]"
            )
        self.unfreeze_epoch = unfreeze_epoch
        self.encoder_learning_rate_scale = encoder_learning_rate_scale

    def on_epoch_started(
        self,
        epoch: int,
        model: Autoencoder,
        optimizer: optim.Optimizer,
        optimizer_type: OptimizerFactory,
        learning_rate: float,
        weight_decay: float,
    ) -> optim.Optimizer:
        if epoch != self.unfreeze_epoch:
            return optimizer
        model.encoder.unfreeze()
        return _differential_optimizer(
            optimizer_type,
            model,
            learning_rate,
            weight_decay,
            self.encoder_learning_rate_scale,
        )


class DifferentialLearningRateStrategy(TransferLearningStrategy):
    def __init__(self, encoder_learning_rate_scale: float = 0.1) -> None:
        if not 0 < encoder_learning_rate_scale <= 1:
            raise ValueError("encoder learning-rate scale must be in (0, 1]")
        self.encoder_learning_rate_scale = encoder_learning_rate_scale

    def initialize(
        self,
        model: Autoencoder,
        optimizer: OptimizerFactory,
        learning_rate: float,
        weight_decay: float,
    ) -> optim.Optimizer:
        model.encoder.unfreeze()
        return _differential_optimizer(
            optimizer,
            model,
            learning_rate,
            weight_decay,
            self.encoder_learning_rate_scale,
        )


def _non_encoder_parameters(model: Autoencoder) -> Iterator[nn.Parameter]:
    encoder_parameter_ids = {id(parameter) for parameter in model.encoder.parameters()}
    return (
        parameter
        for parameter in model.parameters()
        if id(parameter) not in encoder_parameter_ids
    )


def _optimizer(
    optimizer: OptimizerFactory,
    parameters: Iterator[nn.Parameter],
    learning_rate: float,
    weight_decay: float,
) -> optim.Optimizer:
    return optimizer(parameters, lr=learning_rate, weight_decay=weight_decay)


def _differential_optimizer(
    optimizer: OptimizerFactory,
    model: Autoencoder,
    learning_rate: float,
    weight_decay: float,
    encoder_learning_rate_scale: float,
) -> optim.Optimizer:
    return optimizer(
        (
            {
                "params": model.encoder.parameters(),
                "lr": learning_rate * encoder_learning_rate_scale,
            },
            {"params": _non_encoder_parameters(model), "lr": learning_rate},
        ),
        weight_decay=weight_decay,
    )
