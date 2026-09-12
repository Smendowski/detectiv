from dataclasses import dataclass

from detectiv.scoring import ReconstructionScorer, WindowToPointScorePropagationStrategy


@dataclass(frozen=True)
class ScoringPlan:
    scorer: ReconstructionScorer
    propagations: tuple[WindowToPointScorePropagationStrategy, ...]

    def __post_init__(self) -> None:
        if not self.propagations:
            raise ValueError("scoring plans require at least one propagation strategy")

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
