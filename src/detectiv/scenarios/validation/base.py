from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ValidationPartition:
    training_indices: NDArray[np.intp]
    validation_indices: NDArray[np.intp]


class ValidationHoldout(ABC):
    @abstractmethod
    def split(self, indices: NDArray[np.intp]) -> ValidationPartition:
        raise NotImplementedError
