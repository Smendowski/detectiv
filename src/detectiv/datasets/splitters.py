from collections.abc import Mapping
from dataclasses import dataclass

from detectiv.data import TemporalSplit, TimeSeries
from detectiv.datasets.time_series import TimeSeriesDataset


@dataclass(frozen=True)
class TemporalBoundary:
    train_end: int
    validation_end: int | None = None

    def __post_init__(self) -> None:
        if self.train_end <= 0:
            raise ValueError("train_end must be positive")
        if self.validation_end is not None and self.validation_end <= self.train_end:
            raise ValueError("validation_end must be greater than train_end")


@dataclass(frozen=True)
class TemporalHoldout:
    test_start: int
    validation_fraction: float

    def __post_init__(self) -> None:
        if self.test_start <= 1:
            raise ValueError("test_start must leave room for training and validation")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")

    def boundary(self) -> TemporalBoundary:
        validation_length = round(self.test_start * self.validation_fraction)
        validation_length = min(max(validation_length, 1), self.test_start - 1)
        return TemporalBoundary(
            train_end=self.test_start - validation_length,
            validation_end=self.test_start,
        )


class TemporalSplitter:
    def __init__(
        self,
        rules: Mapping[str, TemporalBoundary | TemporalHoldout],
    ) -> None:
        if not rules:
            raise ValueError("boundaries must not be empty")
        self.rules = dict(rules)

    def split(self, dataset: TimeSeriesDataset) -> TemporalSplit[TimeSeriesDataset]:
        if set(dataset.series_ids) != set(self.rules):
            raise ValueError("boundaries must match dataset series IDs")

        splits = {
            series_id: dataset[series_id].split(
                train_end=self._boundary(rule).train_end,
                validation_end=self._boundary(rule).validation_end,
            )
            for series_id, rule in self.rules.items()
        }
        validation = None
        if any(item.validation is not None for item in splits.values()):
            validation_series: dict[str, TimeSeries] = {}
            for series_id, item in splits.items():
                if item.validation is None:
                    raise ValueError(
                        "all series must either include validation or omit it"
                    )
                validation_series[series_id] = item.validation
            validation = TimeSeriesDataset(
                f"{dataset.dataset_id}:validation",
                validation_series,
                metadata=dataset.metadata,
            )

        return TemporalSplit(
            train=TimeSeriesDataset(
                f"{dataset.dataset_id}:train",
                {series_id: item.train for series_id, item in splits.items()},
                metadata=dataset.metadata,
            ),
            validation=validation,
            test=TimeSeriesDataset(
                f"{dataset.dataset_id}:test",
                {series_id: item.test for series_id, item in splits.items()},
                metadata=dataset.metadata,
            ),
        )

    @staticmethod
    def _boundary(rule: TemporalBoundary | TemporalHoldout) -> TemporalBoundary:
        if isinstance(rule, TemporalHoldout):
            return rule.boundary()
        return rule
