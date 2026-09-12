from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, optim
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader

from detectiv.datasets import ImageDataset, TorchImageDataset
from detectiv.losses import MeanSquaredReconstructionLoss
from detectiv.models.autoencoders.model import Autoencoder
from detectiv.models.autoencoders.transfer_learning import (
    OptimizerFactory,
    TransferLearningStrategy,
)
from detectiv.models.runtime import resolve_device

Scheduler = LRScheduler | ReduceLROnPlateau
SchedulerFactory = Callable[[optim.Optimizer], Scheduler]


@dataclass(frozen=True)
class AutoencoderTrainer:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: str = "auto"
    seed: int | None = None
    optimizer: OptimizerFactory = torch.optim.AdamW
    scheduler_factory: SchedulerFactory | None = None
    transfer_strategy: TransferLearningStrategy = field(
        default_factory=TransferLearningStrategy
    )
    loss: MeanSquaredReconstructionLoss = field(
        default_factory=MeanSquaredReconstructionLoss
    )
    early_stopping_patience: int | None = None
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError(
                "learning_rate must be positive and weight_decay non-negative"
            )
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience <= 0
        ):
            raise ValueError("early_stopping_patience must be positive")
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative")

    def fit(
        self,
        model: Autoencoder,
        images: ImageDataset,
        indices: Sequence[int] | NDArray[np.intp],
        *,
        validation: ImageDataset | None = None,
        validation_indices: Sequence[int] | NDArray[np.intp] | None = None,
        on_epoch_finished: Callable[[int, float], None] | None = None,
    ) -> "TrainingHistory":
        dataset = TorchImageDataset(images, indices)
        if not len(dataset):
            raise ValueError("at least one training image is required")
        validation_dataset = self._validation_dataset(validation, validation_indices)
        if self.early_stopping_patience is not None and validation_dataset is None:
            raise ValueError("early stopping requires validation images")
        generator = None
        if self.seed is not None:
            generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        device = resolve_device(self.device)
        model.to(device)
        optimizer = self.transfer_strategy.initialize(
            model,
            self.optimizer,
            self.learning_rate,
            self.weight_decay,
        )
        scheduler = self._scheduler(optimizer, validation_dataset)
        losses: list[float] = []
        validation_losses: list[float] = []
        best_loss = float("inf")
        best_state: dict[str, Tensor] | None = None
        epochs_without_improvement = 0
        model.train()
        for epoch in range(self.epochs):
            updated_optimizer = self.transfer_strategy.on_epoch_started(
                epoch,
                model,
                optimizer,
                self.optimizer,
                self.learning_rate,
                self.weight_decay,
            )
            if updated_optimizer is not optimizer:
                optimizer = updated_optimizer
                scheduler = self._scheduler(optimizer, validation_dataset)
            loss_sum = 0.0
            n_images = 0
            for batch in loader:
                batch = cast(Tensor, batch).to(device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction = model(batch)
                reconstruction_loss = self.loss(reconstruction, batch)
                torch.autograd.backward(reconstruction_loss)
                optimizer.step()
                loss_sum += float(reconstruction_loss.detach()) * len(batch)
                n_images += len(batch)
            epoch_loss = loss_sum / n_images
            losses.append(epoch_loss)
            if on_epoch_finished is not None:
                on_epoch_finished(epoch, epoch_loss)
            if validation_dataset is None:
                self._step_scheduler(scheduler)
                continue
            validation_loss = self._loss(model, validation_dataset, device)
            validation_losses.append(validation_loss)
            self._step_scheduler(scheduler, validation_loss)
            if validation_loss < best_loss - self.min_delta:
                best_loss = validation_loss
                best_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if (
                    self.early_stopping_patience is not None
                    and epochs_without_improvement >= self.early_stopping_patience
                ):
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return TrainingHistory(tuple(losses), tuple(validation_losses))

    def _scheduler(
        self,
        optimizer: optim.Optimizer,
        validation: TorchImageDataset | None,
    ) -> Scheduler | None:
        if self.scheduler_factory is None:
            return None
        scheduler = self.scheduler_factory(optimizer)
        if validation is None and isinstance(scheduler, ReduceLROnPlateau):
            raise ValueError("ReduceLROnPlateau requires validation images")
        return scheduler

    @staticmethod
    def _step_scheduler(
        scheduler: Scheduler | None,
        validation_loss: float | None = None,
    ) -> None:
        if scheduler is None:
            return
        if isinstance(scheduler, ReduceLROnPlateau):
            if validation_loss is None:
                raise ValueError("ReduceLROnPlateau requires a validation loss")
            scheduler.step(validation_loss)
            return
        scheduler.step()

    def _validation_dataset(
        self,
        images: ImageDataset | None,
        indices: Sequence[int] | NDArray[np.intp] | None,
    ) -> TorchImageDataset | None:
        if images is None:
            if indices is not None:
                raise ValueError("validation indices require validation images")
            return None
        return TorchImageDataset(images, indices)

    def _loss(
        self,
        model: Autoencoder,
        images: TorchImageDataset,
        device: torch.device,
    ) -> float:
        was_training = model.training
        model.eval()
        loss_sum = 0.0
        with torch.no_grad():
            for batch in DataLoader(images, batch_size=self.batch_size):
                batch = cast(Tensor, batch).to(device)
                loss_sum += float(self.loss(model(batch), batch)) * len(batch)
        model.train(was_training)
        return loss_sum / len(images)


@dataclass(frozen=True)
class TrainingHistory:
    training_losses: tuple[float, ...]
    validation_losses: tuple[float, ...] = ()
