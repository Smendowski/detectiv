import numpy as np
from numpy.typing import NDArray

from detectiv.protocols.validation.base import ValidationHoldout, ValidationPartition


class RandomHoldout(ValidationHoldout):
    """Create a reproducible random validation subset of selected images."""

    def __init__(self, *, fraction: float = 0.2, seed: int = 42) -> None:
        if not 0 < fraction < 1:
            raise ValueError("fraction must be between zero and one")
        self.fraction = fraction
        self.seed = seed

    @property
    def requires_non_overlapping_windows(self) -> bool:
        """Require non-overlap to prevent source-point leakage."""
        return True

    def split(self, indices: NDArray[np.intp]) -> ValidationPartition:
        """Split selected indices into deterministic random subsets."""
        selected = np.asarray(indices, dtype=np.intp)
        if selected.ndim != 1 or len(selected) < 2:
            raise ValueError("random holdout requires at least two selected images")
        validation_size = round(len(selected) * self.fraction)
        validation_size = min(max(validation_size, 1), len(selected) - 1)
        order = np.random.default_rng(self.seed).permutation(len(selected))
        training_size = len(selected) - validation_size
        return ValidationPartition(
            selected[order[:training_size]],
            selected[order[training_size:]],
        )
