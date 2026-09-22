import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from detectiv.benchmarks.tsb_ad.loader import TSBADCsvLoader
from detectiv.time_series import TemporalBoundary, TimeSeries


class TSBADCollection(StrEnum):
    """Identify the TSB-AD collection by its expected dimensionality."""

    UNIVARIATE = "TSB-AD-U"
    MULTIVARIATE = "TSB-AD-M"


@dataclass(frozen=True)
class TSBADDataset:
    """One TSB-AD CSV series and its Detectiv temporal boundary."""

    series: TimeSeries
    boundary: TemporalBoundary


@dataclass(frozen=True)
class _SeriesFile:
    path: Path
    source_group: str
    train_end: int


class TSBADCollectionLoader:
    """Lazily load independent TSB-AD CSV run units from local collections."""

    def __init__(self, data_directory: Path) -> None:
        if not data_directory.is_dir():
            raise FileNotFoundError(
                f"TSB-AD data directory does not exist: {data_directory}"
            )
        self.data_directory = data_directory
        self._csv_loader = TSBADCsvLoader()

    def iter_collection(
        self,
        collection: TSBADCollection | str,
        *,
        source_group: str | None = None,
    ) -> Iterator[TSBADDataset]:
        """Yield one CSV at a time in deterministic filename order.

        Args:
            collection: Collection enum or its string value.
            source_group: Optional exact source group such as ``NAB`` or ``SMD``.

        Returns:
            Lazy iterator of independent series run units in filename order.

        Raises:
            ValueError: If the collection value, source group, filename, CSV,
                temporal boundary, or dimensionality is invalid.
            FileNotFoundError: If the collection directory does not exist.
        """
        collection = TSBADCollection(collection)
        if source_group == "":
            raise ValueError("source_group must not be empty")
        for item in self._collection_files(collection):
            if source_group is None or item.source_group == source_group:
                yield self._load(collection, item)

    def source_groups(self, collection: TSBADCollection | str) -> tuple[str, ...]:
        """Return source metadata groups in deterministic filename order.

        Args:
            collection: Collection enum or its string value.

        Returns:
            Unique source-group names in first-filename order.
        """
        collection = TSBADCollection(collection)
        return tuple(
            dict.fromkeys(
                item.source_group for item in self._collection_files(collection)
            )
        )

    def load_series(
        self, collection: TSBADCollection | str, series_id: str
    ) -> TSBADDataset:
        """Load the CSV whose filename stem exactly matches ``series_id``.

        Args:
            collection: Collection enum or its string value.
            series_id: Exact CSV filename stem.

        Returns:
            The independent series and its temporal boundary.

        Raises:
            ValueError: If the collection value, ID, filename, CSV, temporal
                boundary, or dimensionality is invalid.
            FileNotFoundError: If the collection directory does not exist.
        """
        collection = TSBADCollection(collection)
        if not series_id:
            raise ValueError("TSB-AD series ID must not be empty")
        item = next(
            (
                candidate
                for candidate in self._collection_files(collection)
                if candidate.path.stem == series_id
            ),
            None,
        )
        if item is None:
            raise ValueError(
                f"TSB-AD collection {collection} does not contain series {series_id!r}"
            )
        return self._load(collection, item)

    def _collection_files(self, collection: TSBADCollection) -> tuple[_SeriesFile, ...]:
        collection_directory = self.data_directory / collection
        if not collection_directory.is_dir():
            raise FileNotFoundError(
                f"TSB-AD collection directory does not exist: {collection_directory}"
            )
        paths = sorted(collection_directory.glob("*.csv"))
        files = tuple(self._parse_file(path) for path in paths)
        if not files:
            raise ValueError(
                f"TSB-AD collection contains no CSV files: {collection_directory}"
            )
        return files

    def _load(self, collection: TSBADCollection, item: _SeriesFile) -> TSBADDataset:
        loaded = self._csv_loader.load(item.path)
        if item.train_end >= loaded.n_timesteps:
            raise ValueError(f"TSB-AD train boundary must lie within {item.path.name}")
        if collection is TSBADCollection.UNIVARIATE and not loaded.is_univariate:
            raise ValueError(f"TSB-AD-U series must be univariate: {item.path.name}")
        if collection is TSBADCollection.MULTIVARIATE and loaded.is_univariate:
            raise ValueError(f"TSB-AD-M series must be multivariate: {item.path.name}")
        series = TimeSeries(
            loaded.values,
            labels=loaded.labels,
            feature_names=loaded.feature_names,
            sampling_rate=loaded.sampling_rate,
            series_id=loaded.series_id,
            metadata={
                "benchmark": "TSB-AD",
                "collection": str(collection),
                "source_group": item.source_group,
                "source_file": item.path.name,
            },
        )
        return TSBADDataset(series, TemporalBoundary(item.train_end))

    @staticmethod
    def _parse_file(path: Path) -> _SeriesFile:
        match = re.fullmatch(
            r"\d+_(?P<source_group>.+?)_id_.+?_tr_(?P<train_end>\d+)_1st_\d+\.csv",
            path.name,
        )
        if match is None:
            raise ValueError(f"invalid TSB-AD filename: {path.name}")
        return _SeriesFile(
            path=path,
            source_group=match["source_group"],
            train_end=int(match["train_end"]),
        )
