import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from detectiv.benchmarks.tsb_ad.loader import load_tsb_ad_csv
from detectiv.time_series import TemporalBoundary, TimeSeriesSplit

_FILENAME_PATTERN = re.compile(
    r"""
    \d+_                         # benchmark sequence number
    (?P<source_group>.+?)_       # source dataset, for example NAB or SMD
    id_.+?_                      # upstream series identifier
    tr_(?P<train_end>\d+)_       # exclusive end of the training prefix
    1st_\d+                      # upstream first-anomaly marker
    \.csv
    """,
    re.VERBOSE,
)


class TSBADCollection(StrEnum):
    """Identify the TSB-AD collection by its expected dimensionality."""

    UNIVARIATE = "TSB-AD-U"
    MULTIVARIATE = "TSB-AD-M"


@dataclass(frozen=True)
class _SeriesFile:
    path: Path
    source_group: str
    train_end: int


class TSBADCollectionLoader:
    """Load independent TSB-AD series from local benchmark collections."""

    def __init__(self, data_directory: Path) -> None:
        if not data_directory.is_dir():
            raise FileNotFoundError(
                f"TSB-AD data directory does not exist: {data_directory}"
            )
        self.data_directory = data_directory

    def iter_collection(
        self,
        collection: TSBADCollection | str,
        *,
        source_group: str | None = None,
    ) -> Iterator[TimeSeriesSplit]:
        """Yield one CSV at a time in deterministic filename order.

        Args:
            collection: Collection enum or its string value.
            source_group: Optional exact source group such as ``NAB`` or ``SMD``.

        Returns:
            Lazy iterator of temporal series splits in filename order.

        Raises:
            ValueError: If the collection value, source group, filename, CSV,
                temporal boundary, or dimensionality is invalid.
            FileNotFoundError: If the collection directory does not exist.
        """
        collection = TSBADCollection(collection)
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
    ) -> TimeSeriesSplit:
        """Load the CSV whose filename stem exactly matches ``series_id``.

        Args:
            collection: Collection enum or its string value.
            series_id: Exact CSV filename stem.

        Returns:
            The independent chronological train and test partitions.

        Raises:
            ValueError: If the collection value, ID, filename, CSV, temporal
                boundary, or dimensionality is invalid.
            FileNotFoundError: If the collection directory does not exist.
        """
        collection = TSBADCollection(collection)
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

    def _load(self, collection: TSBADCollection, item: _SeriesFile) -> TimeSeriesSplit:
        series = load_tsb_ad_csv(item.path)

        if collection is TSBADCollection.UNIVARIATE and not series.is_univariate:
            raise ValueError(f"TSB-AD-U series must be univariate: {item.path.name}")

        if collection is TSBADCollection.MULTIVARIATE and series.is_univariate:
            raise ValueError(f"TSB-AD-M series must be multivariate: {item.path.name}")

        return series.split(TemporalBoundary(item.train_end))

    @staticmethod
    def _parse_file(path: Path) -> _SeriesFile:
        match = _FILENAME_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid TSB-AD filename: {path.name}")
        return _SeriesFile(
            path=path,
            source_group=match["source_group"],
            train_end=int(match["train_end"]),
        )
