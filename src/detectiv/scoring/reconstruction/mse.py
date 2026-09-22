from collections.abc import Iterator
from typing import cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from detectiv.images import ImageDataset, TorchImageDataset
from detectiv.losses import MeanSquaredReconstructionLoss
from detectiv.models.autoencoders.model import Autoencoder
from detectiv.models.runtime import ComputeDevice, resolve_device
from detectiv.scoring.base import ReconstructionScorer
from detectiv.scoring.window_scores import WindowScoreBatch
from detectiv.ts2i import DataLoaderSettings


class MeanSquaredReconstructionError(ReconstructionScorer):
    """Shared batched inference for mean-squared reconstruction scorers."""

    def __init__(
        self,
        *,
        batch_size: int = 32,
        device: ComputeDevice | str = ComputeDevice.AUTO,
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
            with torch.inference_mode():
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
    """Reduce mean-squared reconstruction error to one score per image window."""

    @property
    def name(self) -> str:
        """Return the stable scorer identifier."""
        return "mean_squared_window"

    def score(self, model: Autoencoder, images: ImageDataset) -> WindowScoreBatch:
        """Return one mean-squared reconstruction score per image window."""
        values = [error.mean(dim=(1, 2, 3)) for error in self._errors(model, images)]
        return WindowScoreBatch(
            torch.cat(values).cpu().numpy() if values else np.empty(0),
            images.window_references,
        )
