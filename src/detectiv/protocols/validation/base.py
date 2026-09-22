from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ValidationPartition:
    """Disjoint training and validation indices within one image dataset."""

    training_indices: NDArray[np.intp]
    validation_indices: NDArray[np.intp]


class ValidationHoldout(ABC):
    """Split selected image indices into training and validation subsets."""

    @property
    def requires_non_overlapping_windows(self) -> bool:
        """Whether the holdout is invalid for overlapping source windows."""
        return False

    @abstractmethod
    def split(self, indices: NDArray[np.intp]) -> ValidationPartition:
        """Return disjoint training and validation subsets of selected indices."""
        raise NotImplementedError
