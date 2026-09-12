import numpy as np
from numpy.typing import NDArray

from detectiv.datasets import ImageDataset
from detectiv.scenarios.training.base import TrainingMode


class SemiSupervisedTraining(TrainingMode):
    def select(self, images: ImageDataset) -> NDArray[np.intp]:
        if images.window_labels is None:
            raise ValueError("semi-supervised training requires window labels")
        indices = np.flatnonzero(~images.window_labels)
        if not len(indices):
            raise ValueError("semi-supervised training requires normal windows")
        return indices
