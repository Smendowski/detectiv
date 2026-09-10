import pytest

from detectiv.windowing import SplitPart, SplitWindowing, WindowSpec


def test_split_windowing_returns_the_spec_for_each_partition() -> None:
    train = WindowSpec(64)
    validation = WindowSpec(32, stride=1)
    test = WindowSpec(16, stride=4)
    windowing = SplitWindowing(train=train, validation=validation, test=test)

    assert windowing.spec_for(SplitPart.TRAIN) is train
    assert windowing.spec_for(SplitPart.VALIDATION) is validation
    assert windowing.spec_for(SplitPart.TEST) is test


def test_split_windowing_rejects_an_unconfigured_validation_spec() -> None:
    windowing = SplitWindowing(train=WindowSpec(64), test=WindowSpec(64))

    with pytest.raises(ValueError, match="not configured"):
        windowing.spec_for(SplitPart.VALIDATION)
