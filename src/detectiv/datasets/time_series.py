from collections.abc import Mapping
from typing import TYPE_CHECKING

from detectiv.data import TemporalSplit, TimeSeries
from detectiv.datasets.base import Dataset

if TYPE_CHECKING:
    from detectiv.datasets.splitters import TemporalSplitter


class TimeSeriesDataset(Dataset[str, TimeSeries]):
    def __init__(
        self,
        dataset_id: str,
        series: Mapping[str, TimeSeries],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(dataset_id, metadata=metadata)
        if not series:
            raise ValueError("series must not be empty")
        if any(not series_id for series_id in series):
            raise ValueError("series IDs must not be empty")
        if any(item.series_id != series_id for series_id, item in series.items()):
            raise ValueError("series IDs must match their mapping keys")

        self.series = dict(series)

    @property
    def series_ids(self) -> tuple[str, ...]:
        return tuple(self.series)

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, series_id: str) -> TimeSeries:
        return self.series[series_id]

    def split(self, splitter: "TemporalSplitter") -> TemporalSplit["TimeSeriesDataset"]:
        return splitter.split(self)
