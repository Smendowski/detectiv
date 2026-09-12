from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from detectiv.datasets import ImageDataset
from detectiv.scenarios.validation import ValidationHoldout


@dataclass(frozen=True)
class TrainingPartition:
    training_indices: NDArray[np.intp]
    validation_images: ImageDataset | None = None
    validation_indices: NDArray[np.intp] | None = None


class TrainingMode(ABC):
    def __init__(self, *, validation_holdout: ValidationHoldout | None = None) -> None:
        self.validation_holdout = validation_holdout

    @abstractmethod
    def select(self, images: ImageDataset) -> NDArray[np.intp]:
        raise NotImplementedError

    def partition(
        self,
        train: ImageDataset,
        validation: ImageDataset | None = None,
    ) -> TrainingPartition:
        training_indices = self.select(train)
        if validation is not None:
            if self.validation_holdout is not None:
                raise ValueError(
                    "validation holdout cannot be combined with temporal validation"
                )
            return TrainingPartition(
                training_indices,
                validation,
                self.select(validation),
            )
        if self.validation_holdout is None:
            return TrainingPartition(training_indices)
        holdout = self.validation_holdout.split(training_indices)
        return TrainingPartition(
            holdout.training_indices,
            train,
            holdout.validation_indices,
        )
