from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalSplit[T]:
    train: T
    test: T
    validation: T | None = None
