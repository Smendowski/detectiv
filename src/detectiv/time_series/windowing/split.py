from enum import StrEnum

from detectiv.time_series.windowing.core import WindowSpec


class SplitPart(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class SplitWindowing:
    def __init__(
        self,
        *,
        train: WindowSpec,
        test: WindowSpec,
        validation: WindowSpec | None = None,
    ) -> None:
        self.train = train
        self.validation = validation
        self.test = test

    def spec_for(self, part: SplitPart | str) -> WindowSpec:
        try:
            part = SplitPart(part)
        except ValueError as error:
            raise ValueError(f"unsupported split part: {part!r}") from error
        if part is SplitPart.TRAIN:
            return self.train
        if part is SplitPart.VALIDATION:
            if self.validation is None:
                raise ValueError("validation window specification is not configured")
            return self.validation
        if part is SplitPart.TEST:
            return self.test
        raise AssertionError("all split parts are handled")
