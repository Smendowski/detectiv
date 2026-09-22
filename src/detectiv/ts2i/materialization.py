from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from csv import DictWriter
from dataclasses import asdict, dataclass
from math import ceil
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

import numpy as np

from detectiv.images import ImageDataset, ImageSource, ImageSplit
from detectiv.images.io import ImageArtifactLabel, ImageFormat
from detectiv.images.io.writers.images import (
    IMAGE_ARTIFACT_MANIFEST_FIELDS,
    image_artifact_path,
    image_artifact_row,
    write_image_artifact_metadata,
)
from detectiv.time_series import TemporalSplit
from detectiv.ts2i.image_source import ProjectedWindowImageSource

MaterializationWorkers = int | Literal["auto", "max"]

_WORKER_SOURCES: tuple[ProjectedWindowImageSource, ...] = ()
_WORKER_DIRECTORIES: tuple[Path, ...] | None = None
_WORKER_LABELS: tuple[np.ndarray | None, ...] = ()


@dataclass(frozen=True)
class DataLoaderSettings:
    """PyTorch data-loader settings recorded with a materialized artifact.

    Args:
        workers: Number of data-loader worker processes.
        prefetch_factor: Batches prefetched by each worker.
        persistent_workers: Keep workers alive between loader iterations.
        pin_memory: Pin loaded image tensors in host memory.
    """

    workers: int = 0
    prefetch_factor: int | None = None
    persistent_workers: bool = False
    pin_memory: bool = False

    def __post_init__(self) -> None:
        if self.workers < 0:
            raise ValueError("DataLoader workers must be non-negative")
        if self.prefetch_factor is not None and self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")
        if self.workers == 0 and (
            self.prefetch_factor is not None or self.persistent_workers
        ):
            raise ValueError(
                "prefetch and persistent workers require DataLoader workers"
            )


@dataclass(frozen=True)
class MaterializationSettings:
    """Output location, worker selection, and measurement settings.

    Args:
        directory: New destination directory for the image artifact.
        workers: Fixed worker count, or ``"auto"``/``"max"`` selection policy.
        warmup: Untimed render measurements before worker selection.
        samples: Timed measurements for each worker candidate.
        improvement_threshold: Relative throughput gain required to add workers.
        loader: Data-loader settings stored with the artifact metadata.
        profile: Whether to write a render profile trace.
    """

    directory: Path
    workers: MaterializationWorkers = 0
    warmup: int = 0
    samples: int = 1
    improvement_threshold: float = 0.05
    loader: DataLoaderSettings = DataLoaderSettings()
    profile: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.workers, int | str) or (
            isinstance(self.workers, str) and self.workers not in {"auto", "max"}
        ):
            raise ValueError("workers must be a non-negative integer, 'auto', or 'max'")
        if isinstance(self.workers, int) and self.workers < 0:
            raise ValueError("materialization workers must be non-negative")
        if self.warmup < 0 or self.samples <= 0:
            raise ValueError("warmup must be non-negative and samples must be positive")
        if self.improvement_threshold < 0:
            raise ValueError("improvement_threshold must be non-negative")


@dataclass(frozen=True)
class CandidateMeasurement:
    """Measured render throughput for one candidate worker count."""

    workers: int
    elapsed_seconds: float
    throughput: float


@dataclass(frozen=True)
class MaterializationReport:
    """Selected rendering configuration and its worker measurements."""

    requested_workers: MaterializationWorkers
    selected_workers: int
    candidates: tuple[CandidateMeasurement, ...]
    fallback_reason: str | None
    loader: DataLoaderSettings
    profile_trace: str | None = None

    def record(self) -> dict[str, object]:
        """Return a serializable representation suitable for run metadata.

        Returns:
            Nested built-in values that can be persisted as run metadata.
        """
        return {
            "requested_workers": self.requested_workers,
            "selected_workers": self.selected_workers,
            "candidates": [asdict(value) for value in self.candidates],
            "fallback_reason": self.fallback_reason,
            "loader": asdict(self.loader),
            "profile_trace": self.profile_trace,
        }


class MaterializedImageSplit(ImageSplit):
    """Materialized image partitions with their rendering report."""

    materialization: MaterializationReport

    def __init__(
        self,
        *,
        train: ImageDataset,
        test: ImageDataset,
        validation: ImageDataset | None,
        materialization: MaterializationReport,
        location: Path,
    ) -> None:
        super().__init__(
            train=train,
            test=test,
            validation=validation,
            provenance={
                "source": "materialized",
                "location": str(location),
                "materialization": materialization.record(),
            },
        )
        object.__setattr__(self, "materialization", materialization)


class NpyImageSource(ImageSource):
    """Image source that lazily reads individual non-pickled NPY files."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self._paths = tuple(paths)

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> np.ndarray:
        return cast(np.ndarray, np.load(self._paths[index], allow_pickle=False))

    def rebase(self, source: Path, destination: Path) -> NpyImageSource:
        """Return an equivalent source rooted at a moved artifact directory.

        Args:
            source: Existing artifact root.
            destination: New artifact root.

        Returns:
            A source with each NPY path rebased to ``destination``.
        """
        return NpyImageSource(
            tuple(destination / path.relative_to(source) for path in self._paths)
        )


def materialize(
    images: TemporalSplit[ImageDataset], settings: MaterializationSettings
) -> tuple[TemporalSplit[ImageDataset], MaterializationReport]:
    """Render images to a new artifact directory and return file-backed datasets.

    Args:
        images: Lazy split datasets to render.
        settings: Output directory and worker-selection configuration.

    Returns:
        Split datasets backed by the published NPY artifact and its report.

    Raises:
        FileExistsError: If the requested output directory already exists.
    """
    if settings.directory.exists():
        raise FileExistsError(
            f"materialization directory already exists: {settings.directory}"
        )
    settings.directory.parent.mkdir(parents=True, exist_ok=True)
    sources = _sources(images)
    candidates = _candidate_counts(settings.workers, sources)
    measurements = tuple(
        _measure(sources, count, settings.warmup, settings.samples)
        for count in candidates
    )
    selected, fallback = _select(settings, measurements)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{settings.directory.name}.", dir=settings.directory.parent
        )
    )
    try:
        report = MaterializationReport(
            settings.workers, selected, measurements, fallback, settings.loader
        )
        result = _write_images(staging, images, selected, report)
        write_image_artifact_metadata(staging, result, ImageFormat.NPY)
        _write_manifest(staging, result)
        profile_written = (
            _profile(result.train, staging / "profile.json")
            if settings.profile
            else False
        )
        trace = str(settings.directory / "profile.json") if profile_written else None
        (staging / "materialization.json").write_text(
            json.dumps(
                MaterializationReport(
                    settings.workers,
                    selected,
                    measurements,
                    fallback,
                    settings.loader,
                    trace,
                ).record(),
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(staging, settings.directory)
        result = _rebase(result, staging, settings.directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    report = MaterializationReport(
        settings.workers, selected, measurements, fallback, settings.loader, trace
    )
    return result, report


def _sources(
    images: TemporalSplit[ImageDataset],
) -> tuple[ProjectedWindowImageSource, ...]:
    datasets = (
        (images.train, images.test)
        if images.validation is None
        else (images.train, images.validation, images.test)
    )
    sources = tuple(dataset.source for dataset in datasets)
    if not all(isinstance(source, ProjectedWindowImageSource) for source in sources):
        raise ValueError("only projected TS2I images can be materialized")
    return sources  # type: ignore[return-value]


def _candidate_counts(
    requested: MaterializationWorkers, sources: Iterable[ProjectedWindowImageSource]
) -> tuple[int, ...]:
    chunks = sum(len(source) for source in sources)
    maximum = min(os.cpu_count() or 1, chunks)
    if isinstance(requested, int):
        if requested > maximum:
            raise ValueError(
                f"materialization workers={requested} exceeds "
                f"{maximum} independent windows"
            )
        return (requested,)
    if requested == "max":
        return (maximum,)
    return tuple(
        dict.fromkeys((0, *(value for value in (1, 2, 4, maximum) if value <= maximum)))
    )


def _measure(
    sources: Sequence[ProjectedWindowImageSource],
    workers: int,
    warmup: int,
    samples: int,
) -> CandidateMeasurement:
    for _ in range(warmup):
        _render(sources, workers)
    elapsed = []
    for _ in range(samples):
        started = perf_counter()
        _render(sources, workers)
        elapsed.append(perf_counter() - started)
    seconds = min(elapsed)
    count = sum(len(source) for source in sources)
    return CandidateMeasurement(
        workers,
        seconds,
        count / seconds if seconds else float("inf"),
    )


def _render(sources: Sequence[ProjectedWindowImageSource], workers: int) -> None:
    jobs = _range_jobs(sources, workers)
    if workers == 0:
        for job in jobs:
            _render_range(sources, job)
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(tuple(sources),),
    ) as executor:
        # Workers discard rendered arrays so benchmarking does not retain or serialize
        # the full materialized dataset back to the parent process.
        for _ in executor.map(_render_worker_range, jobs):
            pass


def _range_jobs(
    sources: Sequence[ProjectedWindowImageSource], workers: int
) -> tuple[tuple[int, int, int], ...]:
    total = sum(len(source) for source in sources)
    # A small multiple keeps IPC bounded while giving workers enough work to balance.
    chunk_size = max(1, ceil(total / max(1, workers * 4)))
    return tuple(
        (source_index, start, min(start + chunk_size, len(source)))
        for source_index, source in enumerate(sources)
        for start in range(0, len(source), chunk_size)
    )


def _initialize_worker(
    sources: tuple[ProjectedWindowImageSource, ...],
    directories: tuple[Path, ...] | None = None,
    labels: tuple[np.ndarray | None, ...] = (),
) -> None:
    global _WORKER_SOURCES, _WORKER_DIRECTORIES, _WORKER_LABELS
    _WORKER_SOURCES = sources
    _WORKER_DIRECTORIES = directories
    _WORKER_LABELS = labels


def _render_range(
    sources: Sequence[ProjectedWindowImageSource], job: tuple[int, int, int]
) -> None:
    source_index, start, stop = job
    source = sources[source_index]
    for index in range(start, stop):
        source[index]


def _render_worker_range(job: tuple[int, int, int]) -> None:
    _render_range(_WORKER_SOURCES, job)


def _write_worker_range(job: tuple[int, int, int]) -> None:
    if _WORKER_DIRECTORIES is None:
        raise RuntimeError("materialization worker was not initialized for writing")
    source_index, start, stop = job
    source = _WORKER_SOURCES[source_index]
    directory = _WORKER_DIRECTORIES[source_index]
    for index in range(start, stop):
        np.save(
            _image_path(directory, _WORKER_LABELS[source_index], index),
            source[index],
            allow_pickle=False,
        )


def _select(
    settings: MaterializationSettings, measurements: Sequence[CandidateMeasurement]
) -> tuple[int, str | None]:
    requested = settings.workers
    if isinstance(requested, int):
        return requested, None
    if requested == "max":
        return measurements[0].workers, None
    baseline = measurements[0]
    winner = min(measurements, key=lambda value: value.elapsed_seconds)
    if winner.workers and winner.elapsed_seconds <= baseline.elapsed_seconds * (
        1 - settings.improvement_threshold
    ):
        return winner.workers, None
    return 0, "no concurrent candidate met the improvement threshold"


def _write_images(
    directory: Path,
    images: TemporalSplit[ImageDataset],
    workers: int,
    report: MaterializationReport,
) -> TemporalSplit[ImageDataset]:
    datasets = _named_datasets(images)
    sources = tuple(
        cast(ProjectedWindowImageSource, dataset.source) for _, dataset in datasets
    )
    directories = tuple(directory / name for name, _ in datasets)
    labels = tuple(dataset.window_labels for _, dataset in datasets)
    for path, split_labels in zip(directories, labels, strict=True):
        for label in _labels(split_labels):
            (path / label).mkdir(parents=True, exist_ok=True)
    jobs = _range_jobs(sources, workers)
    if workers == 0:
        for job in jobs:
            _write_range(sources, directories, labels, job)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(sources, directories, labels),
        ) as executor:
            # map preserves range order while worker writes remain isolated by index.
            for _ in executor.map(_write_worker_range, jobs):
                pass

    _validate_written_images(datasets, directory)

    result: dict[str, ImageDataset] = {}
    for name, dataset in datasets:
        paths = tuple(
            _image_path(directory / name, dataset.window_labels, index)
            for index in range(len(dataset))
        )
        metadata = dict(dataset.metadata)
        metadata["ts2i_materialized"] = True
        metadata["ts2i_performance"] = report.record()
        result[name] = ImageDataset(
            dataset.dataset_id,
            image_shape=dataset.image_shape,
            window_references=dataset.window_references,
            source=NpyImageSource(paths),
            window_labels=dataset.window_labels,
            series_lengths=dataset.series_lengths,
            point_labels=dataset.point_labels,
            metadata=metadata,
        )
    if "validation" in result:
        return TemporalSplit(
            train=result["train"], test=result["test"], validation=result["validation"]
        )
    return TemporalSplit(train=result["train"], test=result["test"])


def _validate_written_images(
    datasets: Sequence[tuple[str, ImageDataset]], directory: Path
) -> None:
    for name, dataset in datasets:
        split_directory = directory / name
        expected_paths = tuple(
            _image_path(split_directory, dataset.window_labels, index)
            for index in range(len(dataset))
        )
        missing = next((path for path in expected_paths if not path.is_file()), None)
        if missing is not None:
            raise RuntimeError(
                f"materialization did not write expected image: {missing}"
            )

        written_count = sum(
            1 for path in split_directory.glob("*/*.npy") if path.is_file()
        )
        if written_count != len(expected_paths):
            raise RuntimeError(
                f"materialization wrote {written_count} {name} images, "
                f"expected {len(expected_paths)}"
            )


def _write_range(
    sources: Sequence[ProjectedWindowImageSource],
    directories: Sequence[Path],
    labels: Sequence[np.ndarray | None],
    job: tuple[int, int, int],
) -> None:
    source_index, start, stop = job
    source = sources[source_index]
    directory = directories[source_index]
    for index in range(start, stop):
        np.save(
            _image_path(directory, labels[source_index], index),
            source[index],
            allow_pickle=False,
        )


def _image_path(directory: Path, labels: np.ndarray | None, index: int) -> Path:
    label = ImageArtifactLabel.from_window_label(
        None if labels is None else bool(labels[index])
    )
    return directory / image_artifact_path("", label, index, ImageFormat.NPY)


def _labels(labels: np.ndarray | None) -> tuple[ImageArtifactLabel, ...]:
    if labels is None:
        return (ImageArtifactLabel.UNLABELED,)
    return tuple(
        label
        for label in ImageArtifactLabel
        if any(
            ImageArtifactLabel.from_window_label(bool(value)) is label
            for value in labels
        )
    )


def _write_manifest(directory: Path, images: TemporalSplit[ImageDataset]) -> None:
    with (directory / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = DictWriter(file, fieldnames=IMAGE_ARTIFACT_MANIFEST_FIELDS)
        writer.writeheader()
        for name, dataset in _named_datasets(images):
            for index in range(len(dataset)):
                writer.writerow(
                    image_artifact_row(name, dataset, index, ImageFormat.NPY)
                )


def _rebase(
    images: TemporalSplit[ImageDataset], source: Path, destination: Path
) -> TemporalSplit[ImageDataset]:
    def rebase(dataset: ImageDataset) -> ImageDataset:
        return ImageDataset(
            dataset.dataset_id,
            image_shape=dataset.image_shape,
            window_references=dataset.window_references,
            source=cast(NpyImageSource, dataset.source).rebase(source, destination),
            window_labels=dataset.window_labels,
            series_lengths=dataset.series_lengths,
            point_labels=dataset.point_labels,
            metadata=dataset.metadata,
        )

    validation = None if images.validation is None else rebase(images.validation)
    return TemporalSplit(
        train=rebase(images.train), test=rebase(images.test), validation=validation
    )


def _named_datasets(
    images: TemporalSplit[ImageDataset],
) -> tuple[tuple[str, ImageDataset], ...]:
    values: list[tuple[str, ImageDataset]] = [("train", images.train)]
    if images.validation is not None:
        values.append(("validation", images.validation))
    values.append(("test", images.test))
    return tuple(values)


def _profile(images: ImageDataset, trace: Path) -> bool:
    try:
        import torch
        from torch.profiler import ProfilerActivity, profile
    except ImportError:
        return False
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as profiler:
        for index in range(min(2, len(images))):
            _ = images[index]
    profiler.export_chrome_trace(str(trace))
    return True
