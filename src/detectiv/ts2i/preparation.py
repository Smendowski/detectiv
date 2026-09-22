from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

import numpy as np

from detectiv.images import ImageDataset, ImageShape
from detectiv.runs import ReproducibilitySettings
from detectiv.time_series import TemporalSplit, TemporalSplitter, TimeSeriesDataset
from detectiv.time_series.preprocessing import TimeSeriesPreprocessor
from detectiv.time_series.windowing import SplitPart, SplitWindowing, WindowSpec
from detectiv.ts2i.image_source import ProjectedWindowImageSource
from detectiv.ts2i.materialization import (
    MaterializationSettings,
    MaterializedImageSplit,
    materialize,
)
from detectiv.ts2i.projection import ProjectionScheme, ProjectionStrategy


@dataclass(frozen=True)
class ImagePreparation:
    """Immutable builder for a split, windowed, projected image dataset.

    Args:
        dataset: Source time-series dataset to turn into images.
    """

    dataset: TimeSeriesDataset
    _splitter: TemporalSplitter | None = None
    _windowing: SplitWindowing | None = None
    _projection: ProjectionStrategy | None = None
    _preprocessor: TimeSeriesPreprocessor | None = None

    def split(self, splitter: TemporalSplitter) -> ImagePreparation:
        """Return a preparation configured with its temporal splitter.

        Args:
            splitter: Strategy that partitions each source series over time.

        Returns:
            A new preparation with ``splitter`` configured.
        """
        return replace(self, _splitter=splitter)

    def window(
        self,
        *,
        train: WindowSpec,
        test: WindowSpec,
        validation: WindowSpec | None = None,
    ) -> ImagePreparation:
        """Return a preparation with per-split window specifications.

        Args:
            train: Window specification for training series.
            test: Window specification for test series.
            validation: Optional window specification for validation series.

        Returns:
            A new preparation with the supplied window specifications.
        """
        return replace(
            self,
            _windowing=SplitWindowing(train=train, validation=validation, test=test),
        )

    def project(self, projection: ProjectionStrategy) -> ImagePreparation:
        """Return a preparation configured with its projection strategy.

        Args:
            projection: Strategy fit on training data to render image channels.

        Returns:
            A new preparation with ``projection`` configured.
        """
        return replace(self, _projection=projection)

    def preprocess(self, preprocessor: TimeSeriesPreprocessor) -> ImagePreparation:
        """Return a preparation configured with split-aware preprocessing.

        Args:
            preprocessor: Transformer fit on training data before windowing.

        Returns:
            A new preparation with ``preprocessor`` configured.
        """
        return replace(self, _preprocessor=preprocessor)

    def build(
        self, size: tuple[int, int], *, seed: int = 0
    ) -> TemporalSplit[ImageDataset]:
        """Fit the pipeline and return lazy image datasets for every split.

        Args:
            size: Output image height and width.
            seed: Seed from which deterministic per-window render seeds derive.

        Returns:
            Train, optional validation, and test image datasets.

        Raises:
            ValueError: If splitting, windowing, or projection was not configured.
        """
        splitter = self._required("splitter", self._splitter)
        windowing = self._required("windowing", self._windowing)
        projection = self._required("projection strategy", self._projection)
        split = self.dataset.split(splitter)
        split = self._preprocess(split, self._preprocessor)
        fitted_projection = projection.fit(split.train)

        validation = None
        if split.validation is not None:
            validation = self._images(
                split.validation,
                windowing.spec_for(SplitPart.VALIDATION),
                fitted_projection,
                size,
                seed,
                SplitPart.VALIDATION,
            )
        return TemporalSplit(
            train=self._images(
                split.train,
                windowing.spec_for(SplitPart.TRAIN),
                fitted_projection,
                size,
                seed,
                SplitPart.TRAIN,
            ),
            validation=validation,
            test=self._images(
                split.test,
                windowing.spec_for(SplitPart.TEST),
                fitted_projection,
                size,
                seed,
                SplitPart.TEST,
            ),
        )

    def inspect(
        self, size: tuple[int, int], *, seed: int = 0
    ) -> ImagePreparationInspection:
        """Build images and summarize their shape, count, and value range.

        Args:
            size: Output image height and width.
            seed: Seed used for deterministic rendering.

        Returns:
            Aggregate and per-series inspection information for each split.
        """
        images = self.build(size, seed=seed)
        return ImagePreparationInspection(
            train=_inspect_images(images.train),
            validation=None
            if images.validation is None
            else _inspect_images(images.validation),
            test=_inspect_images(images.test),
        )

    def materialize(
        self,
        size: tuple[int, int],
        settings: MaterializationSettings,
        *,
        reproducibility: ReproducibilitySettings | None = None,
    ) -> MaterializedImageSplit:
        """Render this preparation once into an atomically published image artifact.

        Args:
            size: Output image height and width.
            settings: Destination and worker configuration for materialization.
            reproducibility: Optional settings that seed deterministic rendering.

        Returns:
            File-backed split datasets carrying their materialization report.
        """
        seed = 0 if reproducibility is None else reproducibility.seed
        images, report = materialize(self.build(size, seed=seed), settings)
        return MaterializedImageSplit(
            train=images.train,
            validation=images.validation,
            test=images.test,
            materialization=report,
            location=settings.directory,
        )

    @staticmethod
    def _preprocess(
        split: TemporalSplit[TimeSeriesDataset],
        preprocessor: TimeSeriesPreprocessor | None = None,
    ) -> TemporalSplit[TimeSeriesDataset]:
        if preprocessor is None:
            return split
        preprocessor.fit(split.train)
        validation = None
        if split.validation is not None:
            validation = preprocessor.transform(split.validation)
        return TemporalSplit(
            train=preprocessor.transform(split.train),
            validation=validation,
            test=preprocessor.transform(split.test),
        )

    @staticmethod
    def _required[T](name: str, value: T | None) -> T:
        if value is None:
            raise ValueError(f"{name} must be configured before building images")
        return value

    @staticmethod
    def _images(
        dataset: TimeSeriesDataset,
        window: WindowSpec,
        projection: ProjectionScheme,
        size: tuple[int, int],
        seed: int,
        split: SplitPart,
    ) -> ImageDataset:
        source = ProjectedWindowImageSource(
            dataset,
            window=window,
            projection=projection,
            size=size,
            seed=seed,
            split=split.value,
        )
        return ImageDataset(
            f"{dataset.dataset_id}:images",
            image_shape=ImageShape(projection.n_channels, *size),
            window_references=source.window_references,
            source=source,
            window_labels=source.window_labels,
            series_lengths={
                series_id: dataset[series_id].n_timesteps
                for series_id in dataset.series_ids
            },
            point_labels=None
            if dataset[dataset.series_ids[0]].labels is None
            else {
                series_id: cast(np.ndarray, dataset[series_id].labels)
                for series_id in dataset.series_ids
                if dataset[series_id].labels is not None
            },
            metadata=dataset.metadata,
        )


@dataclass(frozen=True)
class ImageSeriesInspection:
    """Image count and value range for one source series."""

    series_length: int
    windows: int
    covered_points: int
    uncovered_points: int
    maximum_coverage: int


@dataclass(frozen=True)
class ImageSplitInspection:
    """Aggregate and per-series inspection details for one temporal split."""

    image_count: int
    image_shape: ImageShape
    series: Mapping[str, ImageSeriesInspection]


@dataclass(frozen=True)
class ImagePreparationInspection:
    """Inspection details for all image dataset partitions."""

    train: ImageSplitInspection
    validation: ImageSplitInspection | None
    test: ImageSplitInspection

    def summary(self) -> str:
        """Return a compact human-readable description of every split.

        Returns:
            One summary line for each configured temporal partition.
        """
        parts = [_split_summary("train", self.train)]
        if self.validation is not None:
            parts.append(_split_summary("validation", self.validation))
        parts.append(_split_summary("test", self.test))
        return "\n".join(parts)


def _inspect_images(images: ImageDataset) -> ImageSplitInspection:
    series: dict[str, ImageSeriesInspection] = {}
    for series_id, series_length in images.series_lengths.items():
        coverage = np.zeros(series_length, dtype=np.intp)
        references = [
            reference
            for reference in images.window_references
            if reference.series_id == series_id
        ]
        for reference in references:
            coverage[reference.start : reference.stop] += 1
        series[series_id] = ImageSeriesInspection(
            series_length=series_length,
            windows=len(references),
            covered_points=int(np.count_nonzero(coverage)),
            uncovered_points=int(np.count_nonzero(coverage == 0)),
            maximum_coverage=int(coverage.max()) if len(coverage) else 0,
        )
    return ImageSplitInspection(
        image_count=len(images),
        image_shape=images.image_shape,
        series=MappingProxyType(series),
    )


def _split_summary(name: str, inspection: ImageSplitInspection) -> str:
    series = ", ".join(
        f"{series_id}: {values.windows} windows, {values.uncovered_points} uncovered"
        for series_id, values in inspection.series.items()
    )
    return (
        f"{name}: {inspection.image_count} images with shape "
        f"{inspection.image_shape.shape} ({series})"
    )
