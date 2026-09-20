import json
from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from detectiv.callbacks import BaseCallback, TimingCallback
from detectiv.images import ImageDataset, ImageShape, ImageSource
from detectiv.models.autoencoders import Autoencoder, AutoencoderTrainer
from detectiv.models.autoencoders.decoders import CNNDecoder
from detectiv.models.autoencoders.encoders import CNNEncoder
from detectiv.protocols import SemiSupervisedTraining
from detectiv.runs import ReproducibilitySettings
from detectiv.scenarios import ReconstructionScenario
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


class FailingFinishedCallback(BaseCallback[object]):
    @property
    def name(self) -> str:
        return "failing_finished"

    def on_run_finished(self, result: object) -> None:
        raise RuntimeError("callback failure")

    def on_run_failed(self, error: BaseException) -> None:
        raise AssertionError("finished callbacks must not receive failure events")


class RecordingCallback(BaseCallback[object]):
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        start_error: BaseException | None = None,
        failure_error: Exception | None = None,
    ) -> None:
        self._name = name
        self.events = events
        self.start_error = start_error
        self.failure_error = failure_error

    @property
    def name(self) -> str:
        return self._name

    def on_run_started(self) -> None:
        self.events.append(f"{self.name}:started")
        if self.start_error is not None:
            raise self.start_error

    def on_epoch_finished(self, event: object) -> None:
        self.events.append(f"{self.name}:epoch")

    def on_run_finished(self, result: object) -> None:
        self.events.append(f"{self.name}:finished")

    def on_run_failed(self, error: BaseException) -> None:
        self.events.append(f"{self.name}:failed:{type(error).__name__}")
        if self.failure_error is not None:
            raise self.failure_error


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
    assert result.resolved_inputs["scenario"] == (
        "detectiv.scenarios.reconstruction.ReconstructionScenario"
    )
    assert result.resolved_inputs["data"] == {
        "train": {
            "dataset_id": "images",
            "image_shape": [1, 4, 4],
            "window_count": 2,
            "series_lengths": {"series": 8},
        },
        "validation": None,
        "test": {
            "dataset_id": "images",
            "image_shape": [1, 4, 4],
            "window_count": 2,
            "series_lengths": {"series": 6},
        },
    }
    trainer = result.resolved_inputs["trainer"]
    scoring_plans = result.resolved_inputs["scoring_plans"]
    assert isinstance(trainer, Mapping)
    assert isinstance(scoring_plans, Sequence)
    assert isinstance(scoring_plans[0], Mapping)
    assert trainer["epochs"] == 1
    assert scoring_plans[0]["name"] == "mean_squared_window"
    json.dumps(result.resolved_inputs)


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


def test_callbacks_run_in_registration_order() -> None:
    images = _images(
        values=[np.zeros((1, 4, 4))],
        references=(WindowReference("series", 0, 4, 4),),
        labels=np.array([False]),
        series_length=4,
    )
    events: list[str] = []
    scenario = _scenario(
        images,
        images,
        callbacks=(
            RecordingCallback("first", events),
            RecordingCallback("second", events),
        ),
    )

    scenario.run()

    assert events == [
        "first:started",
        "second:started",
        "first:epoch",
        "second:epoch",
        "first:finished",
        "second:finished",
    ]


def test_start_failure_notifies_only_previously_started_callbacks() -> None:
    images = _images(
        values=[np.zeros((1, 4, 4))],
        references=(WindowReference("series", 0, 4, 4),),
        labels=np.array([False]),
        series_length=4,
    )
    events: list[str] = []
    scenario = _scenario(
        images,
        images,
        callbacks=(
            RecordingCallback("first", events),
            RecordingCallback(
                "failing", events, start_error=RuntimeError("start failed")
            ),
            RecordingCallback("last", events),
        ),
    )

    with pytest.raises(RuntimeError, match="start failed"):
        scenario.run()

    assert events == ["first:started", "failing:started", "first:failed:RuntimeError"]


def test_failure_notification_does_not_replace_the_primary_error() -> None:
    images = _images(
        values=[np.zeros((1, 4, 4))],
        references=(WindowReference("series", 0, 4, 4),),
        labels=np.array([False]),
        series_length=4,
    )
    empty = ImageDataset(
        "empty",
        image_shape=ImageShape(1, 4, 4),
        window_references=(),
        source=ArrayImageSource([]),
    )
    events: list[str] = []
    scenario = _scenario(
        images,
        empty,
        callbacks=(
            RecordingCallback(
                "failing", events, failure_error=RuntimeError("notification failed")
            ),
        ),
    )

    with pytest.raises(ValueError, match="test images") as error:
        scenario.run()

    assert events == ["failing:started", "failing:failed:ValueError"]
    assert any("notification failed" in note for note in error.value.__notes__)


def test_interruption_is_not_reported_as_a_run_failure() -> None:
    images = _images(
        values=[np.zeros((1, 4, 4))],
        references=(WindowReference("series", 0, 4, 4),),
        labels=np.array([False]),
        series_length=4,
    )
    events: list[str] = []
    scenario = _scenario(
        images,
        images,
        callbacks=(
            RecordingCallback("interrupting", events, start_error=KeyboardInterrupt()),
        ),
    )

    with pytest.raises(KeyboardInterrupt):
        scenario.run()

    assert events == ["interrupting:started"]


def test_scenario_records_reproducibility_settings() -> None:
    images = _images(
        values=[np.zeros((1, 4, 4))],
        references=(WindowReference("series", 0, 4, 4),),
        labels=np.array([False]),
        series_length=4,
    )
    settings = ReproducibilitySettings(seed=42, deterministic_algorithms=True)
    settings.apply()
    scenario = _scenario(images, images, reproducibility=settings)

    result = scenario.run()

    assert result.reproducibility["seed"] == 42
    assert result.reproducibility["deterministic_algorithms"] is True
    assert isinstance(result.reproducibility["device"], str)


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


def _scenario(
    train: ImageDataset,
    test: ImageDataset,
    *,
    callbacks: tuple[BaseCallback[object], ...] = (),
    reproducibility: ReproducibilitySettings | None = None,
) -> ReconstructionScenario:
    return ReconstructionScenario(
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
        callbacks=callbacks,
        trainer=AutoencoderTrainer(epochs=1, batch_size=1, shuffle_seed=7),
        reproducibility=reproducibility,
    )
