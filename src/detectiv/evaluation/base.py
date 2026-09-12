from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np


class PointScoreEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self, point_scores: np.ndarray, labels: np.ndarray
    ) -> Mapping[str, float]:
        raise NotImplementedError
