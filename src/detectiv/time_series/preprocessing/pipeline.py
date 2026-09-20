from __future__ import annotations

from collections.abc import Sequence

from detectiv.time_series import TimeSeriesDataset
from detectiv.time_series.preprocessing.base import TimeSeriesPreprocessor


class PreprocessingPipeline(TimeSeriesPreprocessor):
    """Compose preprocessing steps that are fitted and applied in declaration order."""

    def __init__(self, steps: Sequence[TimeSeriesPreprocessor]) -> None:
        """Create a pipeline from at least one preprocessing step."""
        if not steps:
            raise ValueError("steps must not be empty")
        self.steps = tuple(steps)

    def fit(self, train: TimeSeriesDataset) -> PreprocessingPipeline:
        """Fit each step on the training output of its preceding steps."""
        transformed = train
        for step in self.steps:
            step.fit(transformed)
            transformed = step.transform(transformed)
        return self

    def transform(self, dataset: TimeSeriesDataset) -> TimeSeriesDataset:
        """Apply each fitted step to a dataset in declaration order."""
        transformed = dataset
        for step in self.steps:
            transformed = step.transform(transformed)
        return transformed
