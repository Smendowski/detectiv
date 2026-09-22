from collections.abc import Callable
from typing import TypeVar

from detectiv.ts2i.transformations.base import TS2ITransformation

TransformationType = TypeVar("TransformationType", bound=type[TS2ITransformation])
_TRANSFORMATIONS: dict[str, type[TS2ITransformation]] = {}


def register_transformation(
    name: str,
) -> Callable[[TransformationType], TransformationType]:
    """Return a decorator that registers a transformation under ``name``.

    Args:
        name: Unique non-empty name used for later construction.

    Returns:
        A class decorator that registers one transformation type.

    Raises:
        ValueError: If ``name`` is empty or already registered.
    """
    if not name:
        raise ValueError("transformation name must not be empty")

    def register(
        transformation: TransformationType,
    ) -> TransformationType:
        if name in _TRANSFORMATIONS:
            raise ValueError(f"transformation is already registered: {name}")
        _TRANSFORMATIONS[name] = transformation
        return transformation

    return register


def create_transformation(
    name: str,
    **parameters: object,
) -> TS2ITransformation:
    """Instantiate the transformation registered under ``name``.

    Args:
        name: Registered transformation name.
        **parameters: Constructor keyword arguments.

    Returns:
        A newly configured transformation.

    Raises:
        ValueError: If no transformation is registered under ``name``.
    """
    try:
        transformation = _TRANSFORMATIONS[name]
    except KeyError as error:
        raise ValueError(f"unknown transformation: {name}") from error
    return transformation(**parameters)


def available_transformations() -> tuple[str, ...]:
    """Return registered transformation names in deterministic order.

    Returns:
        Alphabetically sorted registered names.
    """
    return tuple(sorted(_TRANSFORMATIONS))
