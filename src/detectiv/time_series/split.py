from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalSplit[T]:
    """Chronological train, optional validation, and test partitions."""

    train: T
    test: T
    validation: T | None = None
