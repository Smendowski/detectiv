from dataclasses import dataclass

from detectiv.scoring.base import ReconstructionScorer
from detectiv.scoring.propagation import (
    PointAssignment,
    PointScoreAggregator,
)


@dataclass(frozen=True)
class PointScoringPlan:
    assignment: PointAssignment
    aggregator: PointScoreAggregator

    @property
    def name(self) -> str:
        return f"{self.assignment.name}_{self.aggregator.name}"


@dataclass(frozen=True)
class ReconstructionScoringPlan:
    scorer: ReconstructionScorer
    point_scoring: tuple[PointScoringPlan, ...]

    def __post_init__(self) -> None:
        if not self.point_scoring:
            raise ValueError("scoring plans require at least one point-scoring plan")
        if len({plan.name for plan in self.point_scoring}) != len(self.point_scoring):
            raise ValueError("point-scoring plan names must be unique")

    @property
    def name(self) -> str:
        return self.scorer.name
