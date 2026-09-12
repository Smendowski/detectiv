import numpy as np

from detectiv.data import TemporalSplit, WindowReference
from detectiv.datasets import ImageDataset, ImageShape, ImageSource
from detectiv.models.autoencoders import Autoencoder, AutoencoderTrainer
from detectiv.models.autoencoders.decoders import CNNDecoder
from detectiv.models.autoencoders.encoders import CNNEncoder
from detectiv.scenarios import (
    ReconstructionScenario,
    ScoringPlan,
    SemiSupervisedTraining,
    TimeCallback,
)
from detectiv.scoring import MeanPropagationStrategy, MeanSquaredWindowError


class ArrayImageSource(ImageSource):
    def __init__(self, values: list[np.ndarray]) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.values[index]


def test_reconstruction_scenario_runs_from_images_to_point_scores() -> None:
    train = _images(
        values=[np.zeros((1, 4, 4)), np.ones((1, 4, 4))],
        references=(
            WindowReference("series", 0, 4, 4),
            WindowReference("series", 4, 8, 4),
        ),
        labels=np.array([False, True]),
        series_length=8,
    )
    test = _images(
        values=[np.zeros((1, 4, 4)), np.ones((1, 4, 4))],
        references=(
            WindowReference("series", 0, 4, 4),
            WindowReference("series", 2, 6, 4),
        ),
        labels=np.array([False, True]),
        series_length=6,
    )
    timer = TimeCallback()
    scenario = ReconstructionScenario(
        images=TemporalSplit(train=train, test=test),
        model=Autoencoder(
            CNNEncoder(1, hidden_channels=(4,)),
            CNNDecoder(4, hidden_channels=(), output_channels=1),
        ),
        training_mode=SemiSupervisedTraining(),
        scoring_plans=(
            ScoringPlan(MeanSquaredWindowError(), (MeanPropagationStrategy(),)),
        ),
        callbacks=(timer,),
        trainer=AutoencoderTrainer(epochs=1, batch_size=1, seed=7),
    )

    result = scenario.run()

    assert len(result.training_losses) == 1
    assert len(result.window_scores["mean_squared_window"].references) == 2
    assert result.point_scores["mean_squared_window"]["mean"]["series"].shape == (6,)
    assert timer.elapsed_seconds is not None


def _images(
    *,
    values: list[np.ndarray],
    references: tuple[WindowReference, ...],
    labels: np.ndarray,
    series_length: int,
) -> ImageDataset:
    return ImageDataset(
        "images",
        image_shape=ImageShape(1, 4, 4),
        window_references=references,
        source=ArrayImageSource(values),
        window_labels=labels,
        series_lengths={"series": series_length},
    )
