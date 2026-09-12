from collections.abc import Callable, Sequence

from torch import nn


def resolve_layer_values(
    values: int | Sequence[int] | None,
    count: int,
    *,
    name: str,
    default: int,
) -> tuple[int, ...]:
    if values is None:
        resolved = (default,) * count
    elif isinstance(values, int):
        resolved = (values,) * count
    else:
        resolved = tuple(values)
    if len(resolved) != count or any(value <= 0 for value in resolved):
        raise ValueError(f"{name} must contain one positive value per layer")
    return resolved


def make_activation(activation: Callable[[], nn.Module]) -> nn.Module:
    layer = activation()
    if not isinstance(layer, nn.Module):
        raise TypeError("activation must create a torch module")
    return layer
