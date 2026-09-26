from operator import index
from typing import SupportsIndex


def integer(value: SupportsIndex, name: str) -> int:
    """Return an integer-like value while rejecting booleans."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def positive_integer(value: SupportsIndex, name: str) -> int:
    """Return a positive integer-like value."""
    message = f"{name} must be a positive integer"
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        value = index(value)
    except TypeError as error:
        raise ValueError(message) from error
    if value <= 0:
        raise ValueError(message)
    return value
