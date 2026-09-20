from enum import StrEnum

from detectiv.time_series.windowing.core import WindowSpec


class SplitPart(StrEnum):
    """Named temporal partitions with independent window specifications."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class SplitWindowing:
    """Window specifications for train, optional validation, and test partitions."""

    def __init__(
        self,
        *,
        train: WindowSpec,
        test: WindowSpec,
        validation: WindowSpec | None = None,
    ) -> None:
        """Create split-specific windowing configuration."""
        self.train = train
        self.validation = validation
        self.test = test

    def spec_for(self, part: SplitPart | str) -> WindowSpec:
        """Return the specification configured for one split partition.

        Raises:
            ValueError: If the part is unknown or validation is not configured.
        """
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
