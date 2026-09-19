import numpy as np
import pytest

from detectiv.callbacks import BaseCallback, TimingCallback
from detectiv.images import ImageDataset, ImageShape, ImageSource
from detectiv.models.autoencoders import Autoencoder, AutoencoderTrainer
from detectiv.models.autoencoders.decoders import CNNDecoder
from detectiv.models.autoencoders.encoders import CNNEncoder
from detectiv.scenarios import ReconstructionScenario, SemiSupervisedTraining
from detectiv.scoring import (
    MeanPointScoreAggregator,
    MeanSquaredWindowReconstructionError,
    PointScoringPlan,
    ReconstructionScoringPlan,
    UniformPointAssignment,
)
from detectiv.time_series import TemporalSplit
from detectiv.time_series.windowing import WindowReference


class ArrayImageSource(ImageSource):
    def __init__(self, values: list[np.ndarray]) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.values[index]


class FailingFinishedCallback(BaseCallback):
    @property
    def name(self) -> str:
        return "failing_finished"

    def on_run_finished(self, result: object) -> None:
        raise RuntimeError("callback failure")

    def on_run_failed(self, error: BaseException) -> None:
        raise AssertionError("finished callbacks must not receive failure events")


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
    timer = TimingCallback()
    scenario = ReconstructionScenario(
        images=TemporalSplit(train=train, test=test),
        model=Autoencoder(
            CNNEncoder(1, hidden_channels=(4,)),
            CNNDecoder(4, hidden_channels=(), output_channels=1),
        ),
        training_mode=SemiSupervisedTraining(),
        scoring_plans=(
            ReconstructionScoringPlan(
                MeanSquaredWindowReconstructionError(),
                (
                    PointScoringPlan(
                        UniformPointAssignment(), MeanPointScoreAggregator()
                    ),
                ),
            ),
        ),
        callbacks=(timer,),
        trainer=AutoencoderTrainer(epochs=1, batch_size=1, shuffle_seed=7),
    )

    inspection = scenario.inspect()

    assert inspection.train_images == 2
    assert inspection.validation_images is None
    assert inspection.test_images == 2
    assert inspection.image_shape == (1, 4, 4)
    assert inspection.scoring_plans == ("mean_squared_window",)
    assert inspection.callbacks == ("timing",)

    result = scenario.run()

    assert len(result.training_losses) == 1
    assert len(result.window_scores["mean_squared_window"].references) == 2
    assert result.point_scores["mean_squared_window"]["uniform_mean"][
        "series"
    ].shape == (6,)
    assert not result.point_scores["mean_squared_window"]["uniform_mean"][
        "series"
    ].flags.writeable
    with pytest.raises(TypeError):
        result.point_scores["other"] = {}  # type: ignore[index]
    assert result.callbacks["timing"] is timer
    assert timer.elapsed_seconds is not None


def test_callback_failure_after_completion_is_not_reported_as_run_failure() -> None:
    images = _images(
        values=[np.zeros((1, 4, 4))],
        references=(WindowReference("series", 0, 4, 4),),
        labels=np.array([False]),
        series_length=4,
    )
    scenario = ReconstructionScenario(
        images=TemporalSplit(train=images, test=images),
        model=Autoencoder(
            CNNEncoder(1, hidden_channels=(4,)),
            CNNDecoder(4, hidden_channels=(), output_channels=1),
        ),
        training_mode=SemiSupervisedTraining(),
        scoring_plans=(
            ReconstructionScoringPlan(
                MeanSquaredWindowReconstructionError(),
                (
                    PointScoringPlan(
                        UniformPointAssignment(), MeanPointScoreAggregator()
                    ),
                ),
            ),
        ),
        callbacks=(FailingFinishedCallback(),),
        trainer=AutoencoderTrainer(epochs=1, batch_size=1, shuffle_seed=7),
    )

    with pytest.raises(RuntimeError, match="callback failure"):
        scenario.run()


def test_reconstruction_scenario_rejects_an_empty_test_dataset() -> None:
    train = _images(
        values=[np.zeros((1, 4, 4))],
        references=(WindowReference("series", 0, 4, 4),),
        labels=np.array([False]),
        series_length=4,
    )
    test = ImageDataset(
        "empty",
        image_shape=ImageShape(1, 4, 4),
        window_references=(),
        source=ArrayImageSource([]),
    )
    scenario = ReconstructionScenario(
        images=TemporalSplit(train=train, test=test),
        model=Autoencoder(
            CNNEncoder(1, hidden_channels=(4,)),
            CNNDecoder(4, hidden_channels=(), output_channels=1),
        ),
        training_mode=SemiSupervisedTraining(),
        scoring_plans=(
            ReconstructionScoringPlan(
                MeanSquaredWindowReconstructionError(),
                (
                    PointScoringPlan(
                        UniformPointAssignment(), MeanPointScoreAggregator()
                    ),
                ),
            ),
        ),
        trainer=AutoencoderTrainer(epochs=1, batch_size=1, shuffle_seed=7),
    )

    with pytest.raises(ValueError, match="test images"):
        scenario.run()


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
