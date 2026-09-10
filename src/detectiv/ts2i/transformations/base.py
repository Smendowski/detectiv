from abc import ABC, abstractmethod
from enum import StrEnum

import numpy as np


class TransformationInput(StrEnum):
    UNIVARIATE = "univariate"
    MULTIVARIATE = "multivariate"


class TS2ITransformation(ABC):
    @property
    @abstractmethod
    def input_kinds(self) -> frozenset[TransformationInput]:
        raise NotImplementedError

    @abstractmethod
    def transform(
        self,
        values: np.ndarray,
        size: tuple[int, int],
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        raise NotImplementedError
