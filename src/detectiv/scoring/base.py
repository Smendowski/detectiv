from abc import ABC, abstractmethod

from detectiv.images import ImageDataset
from detectiv.models.autoencoders.model import Autoencoder
from detectiv.scoring.window_scores import WindowEvidenceBatch


class ReconstructionScorer(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def score(self, model: Autoencoder, images: ImageDataset) -> WindowEvidenceBatch:
        raise NotImplementedError
