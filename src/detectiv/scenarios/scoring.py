from dataclasses import dataclass

from detectiv.scoring import (
    PointAssignment,
    PointScoreAggregator,
    ReconstructionScorer,
)


@dataclass(frozen=True)
class PointScoringPlan:
    assignment: PointAssignment
    aggregator: PointScoreAggregator

    @property
    def name(self) -> str:
        return f"{self.assignment.name}_{self.aggregator.name}"


@dataclass(frozen=True)
class ScoringPlan:
    scorer: ReconstructionScorer
    point_scoring: tuple[PointScoringPlan, ...]

    def __post_init__(self) -> None:
        if not self.point_scoring:
            raise ValueError("scoring plans require at least one point-scoring plan")
        if len({plan.name for plan in self.point_scoring}) != len(self.point_scoring):
            raise ValueError("point-scoring plan names must be unique")

    @property
    def name(self) -> str:
        return _identifier(self.scorer)


def _identifier(value: object) -> str:
    name = type(value).__name__
    for suffix in ("Error", "Strategy"):
        name = name.removesuffix(suffix)
    words: list[str] = []
    for character in name:
        if character.isupper() and words:
            words.append("_")
        words.append(character.lower())
    return "".join(words)
