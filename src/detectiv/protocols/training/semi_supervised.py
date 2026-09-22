import numpy as np
from numpy.typing import NDArray

from detectiv.images import ImageDataset
from detectiv.protocols.training.base import TrainingMode


class SemiSupervisedTraining(TrainingMode):
    """Train only on windows labeled as normal."""

    def select(self, images: ImageDataset) -> NDArray[np.intp]:
        """Return indices whose binary window label is normal."""
        if images.window_labels is None:
            raise ValueError("semi-supervised training requires window labels")

        indices = np.flatnonzero(~images.window_labels)
        if not len(indices):
            raise ValueError("semi-supervised training requires normal windows")

        return indices
