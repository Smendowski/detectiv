import numpy as np
import torch
from numpy.typing import NDArray

from detectiv.scenarios.validation.base import ValidationHoldout, ValidationPartition


class RandomHoldout(ValidationHoldout):
    def __init__(self, *, fraction: float = 0.2, seed: int = 42) -> None:
        if not 0 < fraction < 1:
            raise ValueError("fraction must be between zero and one")
        self.fraction = fraction
        self.seed = seed

    def split(self, indices: NDArray[np.intp]) -> ValidationPartition:
        selected = np.asarray(indices, dtype=np.intp)
        if selected.ndim != 1 or len(selected) < 2:
            raise ValueError("random holdout requires at least two selected images")
        validation_size = round(len(selected) * self.fraction)
        validation_size = min(max(validation_size, 1), len(selected) - 1)
        order = torch.randperm(
            len(selected), generator=torch.Generator().manual_seed(self.seed)
        ).numpy()
        training_size = len(selected) - validation_size
        return ValidationPartition(
            selected[order[:training_size]],
            selected[order[training_size:]],
        )
