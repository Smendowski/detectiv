from collections.abc import Callable
from typing import TypeVar

from detectiv.ts2i.transformations.base import TS2ITransformation

TransformationType = TypeVar("TransformationType", bound=type[TS2ITransformation])
_TRANSFORMATIONS: dict[str, type[TS2ITransformation]] = {}


def register_transformation(
    name: str,
) -> Callable[[TransformationType], TransformationType]:
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
    try:
        transformation = _TRANSFORMATIONS[name]
    except KeyError as error:
        raise ValueError(f"unknown transformation: {name}") from error
    return transformation(**parameters)


def available_transformations() -> tuple[str, ...]:
    return tuple(sorted(_TRANSFORMATIONS))
