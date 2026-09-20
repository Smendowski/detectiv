import platform
import random
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import torch

from detectiv.runs.artifacts import JSONValue


@dataclass(frozen=True)
class ReproducibilitySettings:
    seed: int = 0
    deterministic_algorithms: bool = True

    def apply(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(self.deterministic_algorithms)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = self.deterministic_algorithms

    def record(self, *, device: str | None = None) -> Mapping[str, JSONValue]:
        return MappingProxyType(
            {
                "seed": self.seed,
                "deterministic_algorithms": self.deterministic_algorithms,
                "torch": torch.__version__,
                "python_implementation": platform.python_implementation(),
                **({"device": device} if device is not None else {}),
            }
        )
