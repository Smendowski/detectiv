from dataclasses import dataclass

from detectiv.utils import integer


@dataclass(frozen=True)
class WindowReference:
    """Validated source interval metadata for one window in a named series."""

    series_id: str
    start: int
    stop: int
    valid_length: int

    def __post_init__(self) -> None:
        start = integer(self.start, "start")
        stop = integer(self.stop, "stop")
        valid_length = integer(self.valid_length, "valid_length")
        if not self.series_id:
            raise ValueError("series_id must not be empty")
        if start < 0 or stop <= start:
            raise ValueError("window bounds must be ordered and non-negative")
        if valid_length <= 0 or stop != start + valid_length:
            raise ValueError("valid_length must match the window bounds")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "valid_length", valid_length)
