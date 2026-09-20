from collections.abc import Iterator
from typing import cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from detectiv.images import ImageDataset, TorchImageDataset
from detectiv.losses import MeanSquaredReconstructionLoss
from detectiv.models.autoencoders.model import Autoencoder
from detectiv.models.runtime import resolve_device
from detectiv.scoring.base import ReconstructionScorer
from detectiv.scoring.window_scores import WindowScoreBatch
from detectiv.ts2i import DataLoaderSettings


class MeanSquaredReconstructionError(ReconstructionScorer):
    def __init__(
        self,
        *,
        batch_size: int = 32,
        device: str = "auto",
        loss: MeanSquaredReconstructionLoss | None = None,
        data_loader: DataLoaderSettings | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.device = device
        self.loss = loss or MeanSquaredReconstructionLoss()
        self.data_loader = data_loader or DataLoaderSettings()

    def _errors(self, model: Autoencoder, images: ImageDataset) -> Iterator[Tensor]:
        device = resolve_device(self.device)
        was_training = model.training
        model.to(device)
        model.eval()
        try:
            with torch.no_grad():
                for batch in DataLoader(
                    TorchImageDataset(images),
                    batch_size=self.batch_size,
                    num_workers=self.data_loader.workers,
                    prefetch_factor=self.data_loader.prefetch_factor,
                    persistent_workers=self.data_loader.persistent_workers,
                    pin_memory=self.data_loader.pin_memory and device.type == "cuda",
                ):
                    batch = cast(Tensor, batch).to(
                        device,
                        non_blocking=self.data_loader.pin_memory
                        and device.type == "cuda",
                    )
                    reconstruction = model.reconstruct(batch)
                    yield self.loss.error(reconstruction, batch)
        finally:
            model.train(was_training)


class MeanSquaredWindowReconstructionError(MeanSquaredReconstructionError):
    @property
    def name(self) -> str:
        return "mean_squared_window"

    def score(self, model: Autoencoder, images: ImageDataset) -> WindowScoreBatch:
        values = [
            error.mean(dim=(1, 2, 3)).cpu().numpy()
            for error in self._errors(model, images)
        ]
        return WindowScoreBatch(
            np.concatenate(values) if values else np.empty(0),
            images.window_references,
        )
