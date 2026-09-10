import numpy as np

from detectiv.data import WindowReference
from detectiv.datasets import ImageSource, TimeSeriesDataset
from detectiv.ts2i.projection import ProjectionStrategy
from detectiv.windowing import WindowLabelingStrategy, WindowSpec


class GeneratedImageSource(ImageSource):
    def __init__(
        self,
        dataset: TimeSeriesDataset,
        *,
        window: WindowSpec,
        projection: ProjectionStrategy,
        size: tuple[int, int],
        seed: int,
    ) -> None:
        self._projection = projection
        self._size = size
        self._seed = seed
        windower = window.windower()
        self._batches = {
            series_id: windower.transform(dataset[series_id])
            for series_id in dataset.series_ids
        }
        self._entries = tuple(
            (series_id, index)
            for series_id in dataset.series_ids
            for index in range(self._batches[series_id].n_windows)
        )
        self.window_references = self._references()
        self.window_labels = self._labels(dataset, window.labeling_strategy)

    def _references(self) -> tuple[WindowReference, ...]:
        references: list[WindowReference] = []
        for series_id, index in self._entries:
            batch = self._batches[series_id]
            assert batch.valid_lengths is not None
            start = int(batch.starts[index])
            valid_length = int(batch.valid_lengths[index])
            references.append(
                WindowReference(
                    series_id=series_id,
                    start=start,
                    stop=start + valid_length,
                    valid_length=valid_length,
                )
            )
        return tuple(references)

    def _labels(
        self,
        dataset: TimeSeriesDataset,
        strategy: "WindowLabelingStrategy",
    ) -> np.ndarray | None:
        labels = tuple(dataset[series_id].labels for series_id in dataset.series_ids)
        if all(label is None for label in labels):
            return None
        if any(label is None for label in labels):
            raise ValueError(
                "all series must provide labels or none may provide labels"
            )
        return np.asarray(
            [
                strategy.label(self._series_labels(dataset, reference))
                for reference in self.window_references
            ],
            dtype=bool,
        )

    @staticmethod
    def _series_labels(
        dataset: TimeSeriesDataset, reference: WindowReference
    ) -> np.ndarray:
        labels = dataset[reference.series_id].labels
        if labels is None:
            raise RuntimeError("labels were validated before window labeling")
        return labels[reference.start : reference.stop]

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> np.ndarray:
        if not -len(self) <= index < len(self):
            raise IndexError("image index out of range")
        series_id, window_index = self._entries[index]
        window = self._batches[series_id].values[window_index]
        rng = np.random.default_rng(np.random.SeedSequence((self._seed, index)))
        return self._projection.render(window, self._size, rng)
