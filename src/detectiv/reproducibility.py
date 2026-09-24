import platform
import random
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import torch

from detectiv.typing import JSONValue


@dataclass(frozen=True)
class ReproducibilitySettings:
    """Deterministic random-generator settings shared across a pipeline."""

    seed: int = 0
    deterministic_algorithms: bool = True

    def apply(self) -> None:
        """Apply the configured seed and Torch determinism settings."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(self.deterministic_algorithms)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = self.deterministic_algorithms

    def record(self, *, device: str | None = None) -> Mapping[str, JSONValue]:
        """Return the JSON-compatible execution configuration.

        Args:
            device: Optional execution device recorded with the settings.

        Returns:
            Immutable reproducibility metadata for an experiment report.
        """
        return MappingProxyType(
            {
                "seed": self.seed,
                "deterministic_algorithms": self.deterministic_algorithms,
                "torch": torch.__version__,
                "python_implementation": platform.python_implementation(),
                **({"device": device} if device is not None else {}),
            }
        )


def configure_reproducibility(
    seed: int = 0, *, deterministic_algorithms: bool = True
) -> ReproducibilitySettings:
    """Configure supported random generators and return the applied settings.

    Args:
        seed: Seed applied to Python, NumPy, Torch, and available CUDA devices.
        deterministic_algorithms: Whether Torch must use deterministic algorithms.

    Returns:
        The applied settings for propagation to the experiment pipeline.
    """
    settings = ReproducibilitySettings(seed, deterministic_algorithms)
    settings.apply()
    return settings
