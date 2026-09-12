from abc import ABC, abstractmethod

from detectiv.datasets import ImageDataset
from detectiv.models.autoencoders import Autoencoder
from detectiv.scoring.window_scores import WindowEvidenceBatch


class ReconstructionScorer(ABC):
    @abstractmethod
    def score(self, model: Autoencoder, images: ImageDataset) -> WindowEvidenceBatch:
        raise NotImplementedError
