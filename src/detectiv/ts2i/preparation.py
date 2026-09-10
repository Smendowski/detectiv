from dataclasses import dataclass, replace

from detectiv.data import TemporalSplit
from detectiv.datasets import (
    ImageDataset,
    ImageShape,
    TemporalSplitter,
    TimeSeriesDataset,
)
from detectiv.ts2i.generated import GeneratedImageSource
from detectiv.ts2i.projection import ProjectionStrategy
from detectiv.windowing import SplitPart, SplitWindowing, WindowSpec


@dataclass(frozen=True)
class ImagePreparation:
    dataset: TimeSeriesDataset
    _splitter: TemporalSplitter | None = None
    _windowing: SplitWindowing | None = None
    _projection: ProjectionStrategy | None = None

    def split(self, splitter: TemporalSplitter) -> "ImagePreparation":
        return replace(self, _splitter=splitter)

    def window(
        self,
        *,
        train: WindowSpec,
        test: WindowSpec,
        validation: WindowSpec | None = None,
    ) -> "ImagePreparation":
        return replace(
            self,
            _windowing=SplitWindowing(train=train, validation=validation, test=test),
        )

    def project(self, projection: ProjectionStrategy) -> "ImagePreparation":
        return replace(self, _projection=projection)

    def build(
        self, size: tuple[int, int], *, seed: int = 0
    ) -> TemporalSplit[ImageDataset]:
        splitter = self._required("splitter", self._splitter)
        windowing = self._required("windowing", self._windowing)
        projection = self._required("projection strategy", self._projection)
        split = self.dataset.split(splitter)
        projection.fit(split.train)

        validation = None
        if split.validation is not None:
            validation = self._images(
                split.validation,
                windowing.spec_for(SplitPart.VALIDATION),
                projection,
                size,
                seed,
            )
        return TemporalSplit(
            train=self._images(
                split.train,
                windowing.spec_for(SplitPart.TRAIN),
                projection,
                size,
                seed,
            ),
            validation=validation,
            test=self._images(
                split.test,
                windowing.spec_for(SplitPart.TEST),
                projection,
                size,
                seed,
            ),
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
        projection: ProjectionStrategy,
        size: tuple[int, int],
        seed: int,
    ) -> ImageDataset:
        source = GeneratedImageSource(
            dataset,
            window=window,
            projection=projection,
            size=size,
            seed=seed,
        )
        return ImageDataset(
            f"{dataset.dataset_id}:images",
            image_shape=ImageShape(projection.n_channels, *size),
            window_references=source.window_references,
            source=source,
            window_labels=source.window_labels,
            metadata=dataset.metadata,
        )
