from enum import StrEnum

from detectiv.windowing.core import WindowSpec


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

    def spec_for(self, part: SplitPart) -> WindowSpec:
        if part is SplitPart.TRAIN:
            return self.train
        if part is SplitPart.VALIDATION:
            if self.validation is None:
                raise ValueError("validation window specification is not configured")
            return self.validation
        return self.test
