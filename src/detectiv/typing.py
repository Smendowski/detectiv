from collections.abc import Mapping, Sequence

type JSONValue = (
    bool | int | float | str | Sequence[JSONValue] | Mapping[str, JSONValue] | None
)
