from __future__ import annotations

from hashlib import blake2b

import numpy as np

from detectiv.images import ImageSize, ImageSource
from detectiv.time_series import TimeSeries
from detectiv.time_series.windowing import (
    WindowLabelingStrategy,
    WindowReference,
    WindowSpec,
)
from detectiv.ts2i.projection import BaseProjectionScheme


class ProjectedWindowImageSource(ImageSource):
    """Lazily render referenced windows from a fitted projection scheme."""

    def __init__(
        self,
        series: TimeSeries,
        *,
        window: WindowSpec,
        projection: BaseProjectionScheme,
        size: ImageSize,
        seed: int,
        split: str,
    ) -> None:
        """Window a series and configure deterministic per-window rendering."""
        self._projection = projection
        self._size = size
        self._seed = seed
        self._split = split

        windower = window.windower()
        if series.series_id is None:
            raise ValueError("series must define series_id")
        self._series_id = series.series_id
        self._batch = windower.transform(series)

        self.window_references = self._references()
        self.window_labels = self._labels(series, window.labeling)

    def _references(self) -> tuple[WindowReference, ...]:
        references: list[WindowReference] = []
        assert self._batch.valid_lengths is not None
        for index in range(self._batch.n_windows):
            start = int(self._batch.starts[index])
            valid_length = int(self._batch.valid_lengths[index])
            references.append(
                WindowReference(
                    series_id=self._series_id,
                    start=start,
                    stop=start + valid_length,
                    valid_length=valid_length,
                )
            )
        return tuple(references)

    def _labels(
        self,
        series: TimeSeries,
        strategy: WindowLabelingStrategy,
    ) -> np.ndarray | None:
        if series.labels is None:
            return None
        return np.asarray(
            [
                strategy.label(series.labels[reference.start : reference.stop])
                for reference in self.window_references
            ],
            dtype=bool,
        )

    def __len__(self) -> int:
        """Return the number of generated windows."""
        return self._batch.n_windows

    def __getitem__(self, index: int) -> np.ndarray:
        """Render one channel-first image with a stable window-specific seed."""
        if not -len(self) <= index < len(self):
            raise IndexError("image index out of range")
        if index < 0:
            index += len(self)
        window = self._batch.values[index]
        rng = np.random.default_rng(self._seed_sequence(self.window_references[index]))
        return self._projection.render(window, self._size, rng=rng)

    def _seed_sequence(self, reference: WindowReference) -> np.random.SeedSequence:
        material = "\0".join(
            (
                str(self._seed),
                self._split,
                reference.series_id,
                str(reference.start),
                str(reference.stop),
                str(reference.valid_length),
            )
        ).encode()
        entropy = int.from_bytes(blake2b(material, digest_size=16).digest(), "little")
        return np.random.SeedSequence(entropy)
