from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from detectiv.time_series.series import TimeSeries
from detectiv.time_series.split import TemporalSplit

if TYPE_CHECKING:
    from detectiv.time_series.splitters import TemporalSplitter


class TimeSeriesDataset:
    def __init__(
        self,
        dataset_id: str,
        series: Mapping[str, TimeSeries],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if not dataset_id:
            raise ValueError("dataset_id must not be empty")
        if not series:
            raise ValueError("series must not be empty")
        if any(not series_id for series_id in series):
            raise ValueError("series IDs must not be empty")
        if any(item.series_id != series_id for series_id, item in series.items()):
            raise ValueError("series IDs must match their mapping keys")
        _validate_feature_schema(series)

        self.dataset_id = dataset_id
        self.metadata = MappingProxyType(dict(metadata or {}))
        self.series = MappingProxyType(dict(series))

    @property
    def series_ids(self) -> tuple[str, ...]:
        return tuple(self.series)

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, series_id: str) -> TimeSeries:
        return self.series[series_id]

    def split(self, splitter: "TemporalSplitter") -> TemporalSplit["TimeSeriesDataset"]:
        return splitter.split(self)


def _validate_feature_schema(series: Mapping[str, TimeSeries]) -> None:
    first_series_id, first_series = next(iter(series.items()))
    n_features = first_series.n_features
    feature_names = first_series.feature_names
    for series_id, item in series.items():
        if item.n_features != n_features:
            raise ValueError(
                f"series {series_id!r} has {item.n_features} features; expected "
                f"{n_features} from {first_series_id!r}"
            )
        if item.feature_names != feature_names:
            raise ValueError(
                f"series {series_id!r} feature names must match those of "
                f"{first_series_id!r}"
            )
