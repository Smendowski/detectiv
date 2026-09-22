from dataclasses import dataclass
from typing import NamedTuple


class ImageSize(NamedTuple):
    """Height and width of a rendered image."""

    height: int
    width: int


@dataclass(frozen=True)
class ImageShape:
    """Channel-first shape shared by every image in a dataset."""

    channels: int
    height: int
    width: int

    def __post_init__(self) -> None:
        if self.channels <= 0 or self.height <= 0 or self.width <= 0:
            raise ValueError("image dimensions must be positive")

    @property
    def size(self) -> ImageSize:
        """Return the spatial dimensions in height-width order."""
        return ImageSize(height=self.height, width=self.width)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the channel-first dimensions.

        Returns:
            Channel, height, and width dimensions.
        """
        return (self.channels, *self.size)
