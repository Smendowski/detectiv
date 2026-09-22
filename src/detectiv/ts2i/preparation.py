from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from detectiv.images import ImageDataset, ImageShape, ImageSize
from detectiv.runs import ReproducibilitySettings
from detectiv.time_series import TemporalSplit, TimeSeries
from detectiv.time_series.preprocessing import TimeSeriesPreprocessor
from detectiv.time_series.windowing import (
    SplitPart,
    WindowedTimeSeriesSplit,
    WindowSpec,
)
from detectiv.ts2i.image_source import ProjectedWindowImageSource
from detectiv.ts2i.materialization import (
    MaterializationSettings,
    MaterializedImageSplit,
    materialize,
)
from detectiv.ts2i.projection import ProjectionScheme, ProjectionStrategy


@dataclass(frozen=True)
class ProjectedImageStage:
    """Projected TS2I stage ready to build, inspect, or materialize images.

    Args:
        source: Windowed temporal series partitions.
        projection: Strategy fitted using the transformed training partition.
    """

    source: WindowedTimeSeriesSplit
    projection: ProjectionStrategy

    def build(self, size: ImageSize, *, seed: int = 0) -> TemporalSplit[ImageDataset]:
        """Fit the pipeline and return lazy image datasets for every split.

        Args:
            size: Output image height and width.
            seed: Seed from which deterministic per-window render seeds derive.

        Returns:
            Train, optional validation, and test image datasets.

        """
        windowing = self.source.windowing
        split = self._preprocess(
            self.source.split,
            self.source.split.preprocessors,
        )
        fitted_projection = self.projection.fit(split.train)

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

    def inspect(self, size: ImageSize, *, seed: int = 0) -> ProjectedImageInspection:
        """Build images and summarize their shape, count, and value range.

        Args:
            size: Output image height and width.
            seed: Seed used for deterministic rendering.

        Returns:
            Aggregate and per-series inspection information for each split.
        """
        images = self.build(size, seed=seed)
        return ProjectedImageInspection(
            train=_inspect_images(images.train),
            validation=None
            if images.validation is None
            else _inspect_images(images.validation),
            test=_inspect_images(images.test),
        )

    def materialize(
        self,
        size: ImageSize,
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
        split: TemporalSplit[TimeSeries],
        preprocessors: tuple[TimeSeriesPreprocessor, ...],
    ) -> TemporalSplit[TimeSeries]:
        for preprocessor in preprocessors:
            preprocessor.fit(split.train)
            train = preprocessor.transform(split.train)
            validation = None
            if split.validation is not None:
                validation = preprocessor.transform(split.validation)
            test = preprocessor.transform(split.test)
            split = TemporalSplit(
                train=train,
                validation=validation,
                test=test,
            )
        return split

    @staticmethod
    def _images(
        series: TimeSeries,
        window: WindowSpec,
        projection: ProjectionScheme,
        size: ImageSize,
        seed: int,
        split: SplitPart,
    ) -> ImageDataset:
        if series.series_id is None:
            raise ValueError("series must define series_id")
        source = ProjectedWindowImageSource(
            series,
            window=window,
            projection=projection,
            size=size,
            seed=seed,
            split=split.value,
        )
        return ImageDataset(
            f"{series.series_id}:{split.value}:images",
            image_shape=ImageShape(
                channels=projection.n_channels,
                height=size.height,
                width=size.width,
            ),
            window_references=source.window_references,
            source=source,
            series_id=series.series_id,
            series_length=series.n_timesteps,
            window_labels=source.window_labels,
            point_labels=series.labels,
            metadata=series.metadata,
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
    """Inspection details for one temporal split."""

    image_count: int
    image_shape: ImageShape
    series_id: str
    series: ImageSeriesInspection


@dataclass(frozen=True)
class ProjectedImageInspection:
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
    coverage = np.zeros(images.series_length, dtype=np.intp)
    for reference in images.window_references:
        coverage[reference.start : reference.stop] += 1
    return ImageSplitInspection(
        image_count=len(images),
        image_shape=images.image_shape,
        series_id=images.series_id,
        series=ImageSeriesInspection(
            series_length=images.series_length,
            windows=len(images),
            covered_points=int(np.count_nonzero(coverage)),
            uncovered_points=int(np.count_nonzero(coverage == 0)),
            maximum_coverage=int(coverage.max()),
        ),
    )


def _split_summary(name: str, inspection: ImageSplitInspection) -> str:
    return (
        f"{name}: {inspection.image_count} images with shape "
        f"{inspection.image_shape.shape} ({inspection.series_id}: "
        f"{inspection.series.windows} windows, "
        f"{inspection.series.uncovered_points} uncovered)"
    )
