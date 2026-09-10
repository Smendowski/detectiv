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


class TemporalSplitter:
    def __init__(self, boundaries: Mapping[str, TemporalBoundary]) -> None:
        if not boundaries:
            raise ValueError("boundaries must not be empty")
        self.boundaries = dict(boundaries)

    def split(self, dataset: TimeSeriesDataset) -> TemporalSplit[TimeSeriesDataset]:
        if set(dataset.series_ids) != set(self.boundaries):
            raise ValueError("boundaries must match dataset series IDs")

        splits = {
            series_id: dataset[series_id].split(
                train_end=boundary.train_end,
                validation_end=boundary.validation_end,
            )
            for series_id, boundary in self.boundaries.items()
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
