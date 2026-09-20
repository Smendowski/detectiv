import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from detectiv.benchmarks.tsb_ad.loader import TSBADCsvLoader
from detectiv.time_series import (
    TemporalBoundary,
    TemporalSplitter,
    TimeSeries,
    TimeSeriesDataset,
)


class TSBADCollection(StrEnum):
    """Identify the TSB-AD collection by its expected dimensionality."""

    UNIVARIATE = "TSB-AD-U"
    MULTIVARIATE = "TSB-AD-M"


@dataclass(frozen=True)
class TSBADDataset:
    """Pair an immutable TSB-AD dataset with its temporal split configuration.

    Args:
        dataset: Loaded time-series dataset for one source dataset.
        splitter: Per-series temporal boundaries for splitting the dataset.
    """

    dataset: TimeSeriesDataset
    splitter: TemporalSplitter


@dataclass(frozen=True)
class _SeriesFile:
    path: Path
    dataset_name: str
    train_end: int


class TSBADCollectionLoader:
    """Load TSB-AD collections stored beneath a local data directory.

    Args:
        data_directory: Directory containing one subdirectory for each TSB-AD
            collection.

    Raises:
        FileNotFoundError: If ``data_directory`` does not exist or is not a
            directory.
    """

    def __init__(self, data_directory: Path) -> None:
        if not data_directory.is_dir():
            raise FileNotFoundError(
                f"TSB-AD data directory does not exist: {data_directory}"
            )
        self.data_directory = data_directory
        self._csv_loader = TSBADCsvLoader()

    def load_collection(self, collection: TSBADCollection) -> tuple[TSBADDataset, ...]:
        """Load every source dataset in a collection.

        Args:
            collection: TSB-AD collection to load.

        Returns:
            Eagerly materialized datasets in deterministic source filename order.

        Raises:
            FileNotFoundError: If the collection directory does not exist.
            ValueError: If the collection is empty, a filename or CSV is
                invalid, a train boundary is outside its series, or a series
                does not match the collection's required dimensionality.
        """
        return tuple(self.iter_collection(collection))

    def iter_collection(self, collection: TSBADCollection) -> Iterator[TSBADDataset]:
        """Yield source datasets from a collection in deterministic order.

        Args:
            collection: TSB-AD collection to iterate.

        Yields:
            One dataset at a time in deterministic source filename order.

        Raises:
            FileNotFoundError: If the collection directory does not exist.
            ValueError: If the collection is empty, a filename or CSV is
                invalid, a train boundary is outside its series, or a series
                does not match the collection's required dimensionality.

        Notes:
            CSV files are loaded only when their dataset is requested from the
            iterator; collection filenames are discovered before the first
            item is yielded.
        """
        grouped: dict[str, list[_SeriesFile]] = defaultdict(list)
        for item in self._collection_files(collection):
            grouped[item.dataset_name].append(item)

        for dataset_name, series_files in grouped.items():
            yield self._dataset(collection, dataset_name, series_files)

    def dataset_names(self, collection: TSBADCollection) -> tuple[str, ...]:
        """Return the source dataset names available in a collection.

        Args:
            collection: TSB-AD collection to inspect.

        Returns:
            Unique names in deterministic source filename order.

        Raises:
            FileNotFoundError: If the collection directory does not exist.
            ValueError: If the collection is empty or contains an invalid
                filename.
        """
        files = self._collection_files(collection)
        return tuple(dict.fromkeys(item.dataset_name for item in files))

    def load_dataset(
        self, collection: TSBADCollection, dataset_name: str
    ) -> TSBADDataset:
        """Load one named source dataset from a collection.

        Args:
            collection: TSB-AD collection containing the source dataset.
            dataset_name: Exact source dataset name encoded in its CSV
                filenames.

        Returns:
            Requested dataset with temporal boundaries parsed from CSV filenames.

        Raises:
            FileNotFoundError: If the collection directory does not exist.
            ValueError: If ``dataset_name`` is empty or absent, the collection
                is empty, a filename or CSV is invalid, a train boundary is
                outside its series, or a series has the wrong dimensionality.
        """
        if not dataset_name:
            raise ValueError("TSB-AD dataset name must not be empty")
        files = tuple(
            item
            for item in self._collection_files(collection)
            if item.dataset_name == dataset_name
        )
        if not files:
            raise ValueError(
                f"TSB-AD collection {collection} does not contain dataset "
                f"{dataset_name!r}"
            )
        return self._dataset(collection, dataset_name, list(files))

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

    def _dataset(
        self,
        collection: TSBADCollection,
        dataset_name: str,
        files: list[_SeriesFile],
    ) -> TSBADDataset:
        series: dict[str, TimeSeries] = {}
        boundaries: dict[str, TemporalBoundary] = {}
        for item in files:
            loaded = self._csv_loader.load(item.path)
            series_id = loaded.series_id
            if series_id is None:
                raise RuntimeError("TSB-AD CSV loader must assign a series ID")
            if item.train_end >= loaded.n_timesteps:
                raise ValueError(
                    f"TSB-AD train boundary must lie within {item.path.name}"
                )
            if collection is TSBADCollection.UNIVARIATE and not loaded.is_univariate:
                raise ValueError(
                    f"TSB-AD-U series must be univariate: {item.path.name}"
                )
            if collection is TSBADCollection.MULTIVARIATE and loaded.is_univariate:
                raise ValueError(
                    f"TSB-AD-M series must be multivariate: {item.path.name}"
                )
            series[series_id] = loaded
            boundaries[series_id] = TemporalBoundary(train_end=item.train_end)

        dataset_id = f"{collection}:{dataset_name}"
        dataset = TimeSeriesDataset(
            dataset_id,
            series,
            metadata={
                "benchmark": "TSB-AD",
                "collection": str(collection),
                "source_dataset": dataset_name,
            },
        )
        return TSBADDataset(dataset, TemporalSplitter(boundaries))

    @staticmethod
    def _parse_file(path: Path) -> _SeriesFile:
        match = re.fullmatch(
            r"\d+_(?P<dataset_name>.+?)_id_.+?_tr_(?P<train_end>\d+)_1st_\d+\.csv",
            path.name,
        )
        if match is None:
            raise ValueError(f"invalid TSB-AD filename: {path.name}")
        return _SeriesFile(
            path=path,
            dataset_name=match["dataset_name"],
            train_end=int(match["train_end"]),
        )
