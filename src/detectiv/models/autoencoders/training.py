from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, optim
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader

from detectiv.images import ImageDataset, TorchImageDataset
from detectiv.losses import MeanSquaredReconstructionLoss
from detectiv.models.autoencoders.model import Autoencoder
from detectiv.models.autoencoders.transfer_learning import (
    OptimizerFactory,
    TransferLearningStrategy,
)
from detectiv.models.runtime import ComputeDevice, resolve_device
from detectiv.runs import TrainingEpochEvent
from detectiv.ts2i import DataLoaderSettings

Scheduler = LRScheduler | ReduceLROnPlateau
SchedulerFactory = Callable[[optim.Optimizer], Scheduler]


@dataclass(frozen=True)
class AutoencoderTrainer:
    """Train an autoencoder with optional validation and transfer learning."""

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: ComputeDevice | str = ComputeDevice.AUTO
    shuffle_seed: int | None = None
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
    restore_best_validation: bool = True
    data_loader: DataLoaderSettings = field(default_factory=DataLoaderSettings)

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
        on_epoch_finished: Callable[[TrainingEpochEvent], None] | None = None,
    ) -> TrainingHistory:
        """Fit a model to selected images and return its training history."""
        dataset = TorchImageDataset(images, indices)
        if not len(dataset):
            raise ValueError("at least one training image is required")

        validation_dataset = self._validation_dataset(validation, validation_indices)
        if validation_dataset is not None and not len(validation_dataset):
            raise ValueError("at least one validation image is required")
        if self.early_stopping_patience is not None and validation_dataset is None:
            raise ValueError("early stopping requires validation images")

        generator = None
        if self.shuffle_seed is not None:
            generator = torch.Generator().manual_seed(self.shuffle_seed)
        device = resolve_device(self.device)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=self.data_loader.workers,
            prefetch_factor=self.data_loader.prefetch_factor,
            persistent_workers=self.data_loader.persistent_workers,
            pin_memory=self.data_loader.pin_memory and device.type == "cuda",
        )
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
        data_wait_seconds = 0.0
        transfer_seconds = 0.0
        validation_seconds = 0.0
        best_loss = float("inf")
        best_epoch: int | None = None
        best_state: dict[str, Tensor] | None = None
        epochs_without_improvement = 0
        model.train()
        for epoch in range(self.epochs):
            started_at = perf_counter()
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
            learning_rates = tuple(
                float(group["lr"]) for group in optimizer.param_groups
            )

            loss_sum = 0.0
            n_images = 0
            batches = iter(loader)
            while True:
                waiting_started = perf_counter()
                try:
                    batch = next(batches)
                except StopIteration:
                    break
                data_wait_seconds += perf_counter() - waiting_started
                transfer_started = perf_counter()
                batch = cast(Tensor, batch).to(
                    device,
                    non_blocking=self.data_loader.pin_memory and device.type == "cuda",
                )
                transfer_seconds += perf_counter() - transfer_started
                optimizer.zero_grad(set_to_none=True)
                reconstruction = model(batch)
                reconstruction_loss = self.loss(reconstruction, batch)
                torch.autograd.backward(reconstruction_loss)
                optimizer.step()
                loss_sum += float(reconstruction_loss.detach()) * len(batch)
                n_images += len(batch)

            epoch_loss = loss_sum / n_images
            losses.append(epoch_loss)

            if validation_dataset is None:
                self._step_scheduler(scheduler)
                if on_epoch_finished is not None:
                    on_epoch_finished(
                        TrainingEpochEvent(
                            epoch,
                            epoch_loss,
                            None,
                            learning_rates,
                            perf_counter() - started_at,
                        )
                    )
                continue

            (
                validation_loss,
                validation_wait,
                validation_transfer,
                validation_elapsed,
            ) = self._loss(model, validation_dataset, device)
            data_wait_seconds += validation_wait
            transfer_seconds += validation_transfer
            validation_seconds += validation_elapsed
            validation_losses.append(validation_loss)
            self._step_scheduler(scheduler, validation_loss)
            if on_epoch_finished is not None:
                on_epoch_finished(
                    TrainingEpochEvent(
                        epoch,
                        epoch_loss,
                        validation_loss,
                        learning_rates,
                        perf_counter() - started_at,
                    )
                )

            if validation_loss < best_loss - self.min_delta:
                best_loss = validation_loss
                best_epoch = epoch
                if self.restore_best_validation:
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
        return TrainingHistory(
            training_losses=tuple(losses),
            validation_losses=tuple(validation_losses),
            best_epoch=best_epoch,
            best_validation_loss=None if best_epoch is None else best_loss,
            device=str(device),
            data_wait_seconds=data_wait_seconds,
            transfer_seconds=transfer_seconds,
            validation_seconds=validation_seconds,
        )

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
    ) -> tuple[float, float, float, float]:
        was_training = model.training
        model.eval()
        loss_sum = 0.0
        data_wait_seconds = 0.0
        transfer_seconds = 0.0
        started = perf_counter()

        with torch.no_grad():
            batches = iter(
                DataLoader(
                    images,
                    batch_size=self.batch_size,
                    num_workers=self.data_loader.workers,
                    prefetch_factor=self.data_loader.prefetch_factor,
                    persistent_workers=self.data_loader.persistent_workers,
                    pin_memory=self.data_loader.pin_memory and device.type == "cuda",
                )
            )
            while True:
                waiting_started = perf_counter()
                try:
                    batch = next(batches)
                except StopIteration:
                    break
                data_wait_seconds += perf_counter() - waiting_started
                transfer_started = perf_counter()
                batch = cast(Tensor, batch).to(
                    device,
                    non_blocking=self.data_loader.pin_memory and device.type == "cuda",
                )
                transfer_seconds += perf_counter() - transfer_started
                loss_sum += float(self.loss(model(batch), batch)) * len(batch)

        model.train(was_training)
        return (
            loss_sum / len(images),
            data_wait_seconds,
            transfer_seconds,
            perf_counter() - started,
        )


@dataclass(frozen=True)
class TrainingHistory:
    """Immutable losses, timing, and best-validation metadata from training."""

    training_losses: tuple[float, ...]
    validation_losses: tuple[float, ...] = ()
    best_epoch: int | None = None
    best_validation_loss: float | None = None
    device: str | None = None
    data_wait_seconds: float = 0.0
    transfer_seconds: float = 0.0
    validation_seconds: float = 0.0
