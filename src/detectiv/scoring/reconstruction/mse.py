from collections.abc import Iterator
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader

from detectiv.datasets import ImageDataset, TorchImageDataset
from detectiv.losses import MeanSquaredReconstructionLoss
from detectiv.models.autoencoders import Autoencoder
from detectiv.models.runtime import resolve_device
from detectiv.scoring.base import ReconstructionScorer
from detectiv.scoring.window_scores import (
    WindowPointScoreBatch,
    WindowSaliencyBatch,
    WindowScoreBatch,
)


class MeanSquaredReconstructionError(ReconstructionScorer):
    def __init__(
        self,
        *,
        batch_size: int = 32,
        device: str = "auto",
        loss: MeanSquaredReconstructionLoss | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.device = device
        self.loss = loss or MeanSquaredReconstructionLoss()

    def _errors(self, model: Autoencoder, images: ImageDataset) -> Iterator[Tensor]:
        device = resolve_device(self.device)
        was_training = model.training
        model.to(device)
        model.eval()
        try:
            with torch.no_grad():
                for batch in DataLoader(
                    TorchImageDataset(images), batch_size=self.batch_size
                ):
                    batch = cast(Tensor, batch).to(device)
                    reconstruction = model.reconstruct(batch)
                    yield self.loss.error(reconstruction, batch)
        finally:
            model.train(was_training)


class MeanSquaredWindowError(MeanSquaredReconstructionError):
    def score(self, model: Autoencoder, images: ImageDataset) -> WindowScoreBatch:
        values = [
            error.mean(dim=(1, 2, 3)).cpu().numpy()
            for error in self._errors(model, images)
        ]
        return WindowScoreBatch(
            np.concatenate(values) if values else np.empty(0),
            images.window_references,
        )


class MeanSquaredTemporalColumnError(MeanSquaredReconstructionError):
    def score(self, model: Autoencoder, images: ImageDataset) -> WindowPointScoreBatch:
        values: list[np.ndarray] = []
        offset = 0
        for error in self._errors(model, images):
            column_errors = error.mean(dim=(1, 2)).cpu().numpy()
            references = images.window_references[offset : offset + len(column_errors)]
            values.extend(
                _resample_columns(columns, reference.valid_length)
                for columns, reference in zip(column_errors, references, strict=True)
            )
            offset += len(column_errors)
        return WindowPointScoreBatch(tuple(values), images.window_references)


class MeanSquaredGradientSaliencyError(MeanSquaredReconstructionError):
    def score(self, model: Autoencoder, images: ImageDataset) -> WindowSaliencyBatch:
        device = resolve_device(self.device)
        was_training = model.training
        model.to(device)
        model.eval()
        values: list[np.ndarray] = []
        saliencies: list[np.ndarray] = []
        offset = 0
        try:
            for batch in DataLoader(
                TorchImageDataset(images), batch_size=self.batch_size
            ):
                batch = cast(Tensor, batch).to(device).requires_grad_(True)
                reconstruction = model.reconstruct(batch)
                error = self.loss.error(reconstruction, batch)
                scores = error.mean(dim=(1, 2, 3))
                gradients = torch.autograd.grad(scores.sum(), batch)[0]
                columns = gradients.abs().mean(dim=(1, 2)).detach().cpu().numpy()
                references = images.window_references[offset : offset + len(scores)]
                values.extend(scores.detach().cpu().numpy())
                saliencies.extend(
                    _resample_columns(column, reference.valid_length)
                    for column, reference in zip(columns, references, strict=True)
                )
                offset += len(scores)
        finally:
            model.train(was_training)
        return WindowSaliencyBatch(
            np.asarray(values), tuple(saliencies), images.window_references
        )


def _resample_columns(values: np.ndarray, length: int) -> NDArray[np.float64]:
    if len(values) == length:
        return np.asarray(values, dtype=np.float64)
    positions = np.linspace(0, len(values) - 1, num=length)
    return np.asarray(
        np.interp(positions, np.arange(len(values)), values), dtype=np.float64
    )
