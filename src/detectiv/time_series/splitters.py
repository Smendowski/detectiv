from dataclasses import dataclass
from operator import index
from typing import SupportsIndex, cast


@dataclass(frozen=True)
class TemporalBoundary:
    """Exclusive training and optional validation boundaries for one series."""

    train_end: int
    validation_end: int | None = None

    def __post_init__(self) -> None:
        train_end = _index_value(self.train_end, "train_end")
        validation_end = (
            None
            if self.validation_end is None
            else _index_value(self.validation_end, "validation_end")
        )
        if train_end <= 0:
            raise ValueError("train_end must be positive")
        if validation_end is not None and validation_end <= train_end:
            raise ValueError("validation_end must be greater than train_end")
        object.__setattr__(self, "train_end", train_end)
        object.__setattr__(self, "validation_end", validation_end)


@dataclass(frozen=True)
class TemporalHoldout:
    """Hold out a trailing test segment and derive its validation boundary."""

    test_start: int
    validation_fraction: float

    def __post_init__(self) -> None:
        test_start = _index_value(self.test_start, "test_start")
        if test_start <= 1:
            raise ValueError("test_start must leave room for training and validation")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")
        object.__setattr__(self, "test_start", test_start)

    def boundary(self) -> TemporalBoundary:
        """Return the concrete temporal boundary represented by this holdout rule."""
        validation_length = round(self.test_start * self.validation_fraction)
        validation_length = min(max(validation_length, 1), self.test_start - 1)
        return TemporalBoundary(
            train_end=self.test_start - validation_length,
            validation_end=self.test_start,
        )


def _index_value(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return index(cast(SupportsIndex, value))
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
