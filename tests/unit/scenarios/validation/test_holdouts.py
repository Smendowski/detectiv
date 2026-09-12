import numpy as np

from detectiv.data import WindowReference
from detectiv.datasets import ImageDataset, ImageShape, ImageSource
from detectiv.scenarios.training import SemiSupervisedTraining
from detectiv.scenarios.validation import RandomHoldout


class ArrayImageSource(ImageSource):
    def __len__(self) -> int:
        return 5

    def __getitem__(self, index: int) -> np.ndarray:
        return np.zeros((1, 2, 2), dtype=np.float32)


def test_random_holdout_is_reproducible_and_preserves_selected_images() -> None:
    holdout = RandomHoldout(fraction=0.25, seed=42)
    selected = np.array([0, 2, 3, 4], dtype=np.intp)

    first = holdout.split(selected)
    second = holdout.split(selected)

    assert np.array_equal(first.training_indices, second.training_indices)
    assert np.array_equal(first.validation_indices, second.validation_indices)
    assert first.validation_indices is not None
    assert set(first.training_indices).isdisjoint(first.validation_indices)
    assert set(first.training_indices) | set(first.validation_indices) == set(selected)


def test_semi_supervised_training_applies_its_validation_holdout() -> None:
    images = ImageDataset(
        "images",
        image_shape=ImageShape(channels=1, height=2, width=2),
        window_references=[
            WindowReference("series", index, index + 2, 2) for index in range(5)
        ],
        source=ArrayImageSource(),
        window_labels=np.array([False, True, False, False, False]),
    )

    partition = SemiSupervisedTraining(
        validation_holdout=RandomHoldout(fraction=0.25, seed=42)
    ).partition(images)

    assert partition.validation_images is images
    assert partition.validation_indices is not None
    assert set(partition.training_indices).isdisjoint(partition.validation_indices)
    assert set(partition.training_indices) | set(partition.validation_indices) == {
        0,
        2,
        3,
        4,
    }
