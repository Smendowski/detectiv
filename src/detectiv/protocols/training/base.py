from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from detectiv.images import ImageDataset
from detectiv.protocols.validation import ValidationHoldout


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
        if validation is not None and self.validation_holdout is not None:
            raise ValueError(
                "validation holdout cannot be combined with temporal validation"
            )
        if validation is not None:
            return TrainingPartition(
                training_indices,
                validation,
                self.select(validation),
            )
        if self.validation_holdout is None:
            return TrainingPartition(training_indices)
        if self.validation_holdout.requires_non_overlapping_windows and _overlap(
            train, training_indices
        ):
            raise ValueError(
                "random validation holdout cannot split overlapping windows; use "
                "temporally separated validation images"
            )
        holdout = self.validation_holdout.split(training_indices)
        return TrainingPartition(
            holdout.training_indices,
            train,
            holdout.validation_indices,
        )


def _overlap(images: ImageDataset, indices: NDArray[np.intp]) -> bool:
    references = sorted(
        (images.window_references[int(index)] for index in indices),
        key=lambda reference: (reference.series_id, reference.start),
    )
    stops: dict[str, int] = {}
    for reference in references:
        if reference.start < stops.get(reference.series_id, 0):
            return True
        stops[reference.series_id] = reference.stop
    return False
