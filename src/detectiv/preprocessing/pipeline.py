from collections.abc import Sequence

from detectiv.datasets import TimeSeriesDataset
from detectiv.preprocessing.base import TimeSeriesPreprocessor


class PreprocessingPipeline(TimeSeriesPreprocessor):
    def __init__(self, steps: Sequence[TimeSeriesPreprocessor]) -> None:
        if not steps:
            raise ValueError("steps must not be empty")
        self.steps = tuple(steps)

    def fit(self, train: TimeSeriesDataset) -> "PreprocessingPipeline":
        transformed = train
        for step in self.steps:
            step.fit(transformed)
            transformed = step.transform(transformed)
        return self

    def transform(self, dataset: TimeSeriesDataset) -> TimeSeriesDataset:
        transformed = dataset
        for step in self.steps:
            transformed = step.transform(transformed)
        return transformed
