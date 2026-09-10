from detectiv.ts2i.transformations import (
    MTF,
    available_transformations,
    create_transformation,
)


def test_registry_creates_canonical_transformations() -> None:
    transformation = create_transformation("MTF")

    assert isinstance(transformation, MTF)
    assert available_transformations() == (
        "GADF",
        "GASF",
        "LG",
        "LP",
        "MTF",
        "MWT",
        "RN",
        "RP",
        "RWT",
        "SG",
        "SPIRAL",
    )
