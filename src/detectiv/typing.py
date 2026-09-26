from collections.abc import Mapping, Sequence

type JSONScalar = bool | int | float | str | None
type JSONValue = JSONScalar | Sequence[JSONValue] | Mapping[str, JSONValue]
