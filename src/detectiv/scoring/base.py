from abc import ABC, abstractmethod

from detectiv.images import ImageDataset
from detectiv.models.autoencoders.model import Autoencoder
from detectiv.scoring.window_scores import WindowEvidenceBatch


class ReconstructionScorer(ABC):
    """Produce window-level anomaly evidence from reconstruction behavior."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable scorer identifier used in result keys."""
        raise NotImplementedError

    @abstractmethod
    def score(self, model: Autoencoder, images: ImageDataset) -> WindowEvidenceBatch:
        """Return evidence aligned with every input image window."""
        raise NotImplementedError
