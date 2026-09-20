from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from detectiv.callbacks import BaseCallback
    from detectiv.models.autoencoders import TrainingHistory
    from detectiv.runs import JSONValue
    from detectiv.scoring import WindowEvidenceBatch


@dataclass(frozen=True)
class ReconstructionScenarioResult:
    window_scores: Mapping[str, WindowEvidenceBatch]
    point_scores: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]
    training: TrainingHistory
    callbacks: Mapping[str, BaseCallback[ReconstructionScenarioResult]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    reproducibility: Mapping[str, JSONValue] = field(
        default_factory=lambda: MappingProxyType({})
    )
    resolved_inputs: Mapping[str, JSONValue] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def training_losses(self) -> tuple[float, ...]:
        return self.training.training_losses

    @property
    def validation_losses(self) -> tuple[float, ...]:
        return self.training.validation_losses
