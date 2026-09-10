from dataclasses import dataclass


@dataclass(frozen=True)
class WindowReference:
    series_id: str
    start: int
    stop: int
    valid_length: int

    def __post_init__(self) -> None:
        if not self.series_id:
            raise ValueError("series_id must not be empty")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("window bounds must be ordered and non-negative")
        if not 0 < self.valid_length <= self.stop - self.start:
            raise ValueError("valid_length must lie within the window bounds")
