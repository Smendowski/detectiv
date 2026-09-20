from dataclasses import dataclass
from operator import index


@dataclass(frozen=True)
class WindowReference:
    """Validated source interval metadata for one window in a named series."""

    series_id: str
    start: int
    stop: int
    valid_length: int

    def __post_init__(self) -> None:
        start = _index_value(self.start, "start")
        stop = _index_value(self.stop, "stop")
        valid_length = _index_value(self.valid_length, "valid_length")
        if not self.series_id:
            raise ValueError("series_id must not be empty")
        if start < 0 or stop <= start:
            raise ValueError("window bounds must be ordered and non-negative")
        if valid_length <= 0 or stop != start + valid_length:
            raise ValueError("valid_length must match the window bounds")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "valid_length", valid_length)


def _index_value(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
