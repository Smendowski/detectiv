import numpy as np
import pytest

from detectiv import ImageDataset, ImageShape, ImageSource, WindowReference


class CountingImageSource(ImageSource):
    def __init__(self, image_shape: ImageShape) -> None:
        self.image_shape = image_shape
        self.calls: list[int] = []

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> np.ndarray:
        self.calls.append(index)
        return np.full(self.image_shape.shape, index, dtype=np.float32)


def test_image_dataset_loads_images_lazily() -> None:
    image_shape = ImageShape(channels=3, height=2, width=2)
    references = [WindowReference("series", 0, 4, 4)]
    source = CountingImageSource(image_shape)

    dataset = ImageDataset(
        "images",
        image_shape=image_shape,
        window_references=references,
        source=source,
    )

    assert source.calls == []
    assert dataset[0].shape == (3, 2, 2)
    assert source.calls == [0]


def test_image_dataset_rejects_images_with_the_wrong_shape() -> None:
    dataset = ImageDataset(
        "images",
        image_shape=ImageShape(channels=3, height=2, width=2),
        window_references=[WindowReference("series", 0, 4, 4)],
        source=CountingImageSource(ImageShape(1, 2, 2)),
    )

    with pytest.raises(ValueError, match="expected"):
        dataset[0]
