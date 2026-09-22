from collections.abc import Mapping
from types import MappingProxyType

from detectiv.images.dataset import ImageDataset
from detectiv.time_series import TemporalSplit


class ImageSplit(TemporalSplit[ImageDataset]):
    """Temporal image partitions with source-neutral provenance."""

    provenance: Mapping[str, object]

    def __init__(
        self,
        *,
        train: ImageDataset,
        test: ImageDataset,
        validation: ImageDataset | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(train=train, test=test, validation=validation)
        object.__setattr__(self, "provenance", MappingProxyType(dict(provenance or {})))
