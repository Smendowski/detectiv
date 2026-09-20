from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from detectiv.images import ImageDataset, ImageShape
from detectiv.time_series import TemporalSplit, TemporalSplitter, TimeSeriesDataset
from detectiv.time_series.preprocessing import TimeSeriesPreprocessor
from detectiv.time_series.windowing import SplitPart, SplitWindowing, WindowSpec
from detectiv.ts2i.image_source import ProjectedWindowImageSource
from detectiv.ts2i.materialization import (
    MaterializationReport,
    MaterializationSettings,
    materialize,
)
from detectiv.ts2i.projection import ProjectionScheme, ProjectionStrategy


@dataclass(frozen=True)
class ImagePreparation:
    dataset: TimeSeriesDataset
    _splitter: TemporalSplitter | None = None
    _windowing: SplitWindowing | None = None
    _projection: ProjectionStrategy | None = None
    _preprocessor: TimeSeriesPreprocessor | None = None

    def split(self, splitter: TemporalSplitter) -> ImagePreparation:
        return replace(self, _splitter=splitter)

    def window(
        self,
        *,
        train: WindowSpec,
        test: WindowSpec,
        validation: WindowSpec | None = None,
    ) -> ImagePreparation:
        return replace(
            self,
            _windowing=SplitWindowing(train=train, validation=validation, test=test),
        )

    def project(self, projection: ProjectionStrategy) -> ImagePreparation:
        return replace(self, _projection=projection)

    def preprocess(self, preprocessor: TimeSeriesPreprocessor) -> ImagePreparation:
        return replace(self, _preprocessor=preprocessor)

    def build(
        self, size: tuple[int, int], *, seed: int = 0
    ) -> TemporalSplit[ImageDataset]:
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
        images = self.build(size, seed=seed)
        return ImagePreparationInspection(
            train=_inspect_images(images.train),
            validation=None
            if images.validation is None
            else _inspect_images(images.validation),
            test=_inspect_images(images.test),
        )

    def materialize(
        self, size: tuple[int, int], settings: MaterializationSettings, *, seed: int = 0
    ) -> tuple[TemporalSplit[ImageDataset], MaterializationReport]:
        """Render this preparation once into an atomically published image artifact."""
        return materialize(self.build(size, seed=seed), settings)

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
            metadata=dataset.metadata,
        )


@dataclass(frozen=True)
class ImageSeriesInspection:
    series_length: int
    windows: int
    covered_points: int
    uncovered_points: int
    maximum_coverage: int


@dataclass(frozen=True)
class ImageSplitInspection:
    image_count: int
    image_shape: ImageShape
    series: Mapping[str, ImageSeriesInspection]


@dataclass(frozen=True)
class ImagePreparationInspection:
    train: ImageSplitInspection
    validation: ImageSplitInspection | None
    test: ImageSplitInspection

    def summary(self) -> str:
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
