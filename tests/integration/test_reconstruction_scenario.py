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
from detectiv.reports import ExperimentReport, ReconstructionReport
from detectiv.reproducibility import ReproducibilitySettings
from detectiv.scenarios import ReconstructionScenario
from detectiv.scoring import (
    MeanPointScoreAggregator,
    MeanSquaredWindowReconstructionError,
    PointScoringPlan,
    ReconstructionScorer,
    ReconstructionScoringPlan,
    UniformPointAssignment,
    WindowScoreBatch,
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


class MisalignedScorer(ReconstructionScorer):
    @property
    def name(self) -> str:
        return "misaligned"

    def score(self, model: Autoencoder, images: ImageDataset) -> WindowScoreBatch:
        assert images.window_labels is None
        assert images.point_labels is None
        return WindowScoreBatch(
            np.zeros(len(images)), tuple(reversed(images.window_references))
        )


class FailingFinishedCallback(BaseCallback[ReconstructionReport]):
    @property
    def name(self) -> str:
        return "failing_finished"

    def on_run_finished(self, result: ReconstructionReport) -> None:
        raise RuntimeError("callback failure")

    def on_run_failed(self, error: BaseException) -> None:
        raise AssertionError("finished callbacks must not receive failure events")


class RecordingCallback(BaseCallback[ReconstructionReport]):
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

    def on_run_finished(self, result: ReconstructionReport) -> None:
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
        point_labels=np.array([0, 0, 1, 1, 0, 0]),
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
    assert isinstance(result, ExperimentReport)
    assert "Scenario: reconstruction" in result.summary()
    assert result.summary().splitlines()[1:4] == [
        "Train images: 2",
        "Validation images: none",
        "Test images: 2",
    ]
    assert len(result.window_scores["mean_squared_window"].references) == 2
    assert result.point_scores["mean_squared_window"]["uniform_mean"].shape == (6,)
    assert not result.point_scores["mean_squared_window"][
        "uniform_mean"
    ].flags.writeable
    assert (
        result.point_scores_for()
        is (result.point_scores["mean_squared_window"]["uniform_mean"])
    )
    assert (
        result.point_scores_for(scoring=scenario.scoring_plans[0])
        is result.point_scores_for()
    )
    with pytest.raises(TypeError):
        result.point_scores["other"] = {}  # type: ignore[index]
    assert not hasattr(result, "callbacks")
    assert result.point_labels is not None
    np.testing.assert_array_equal(result.point_labels, [0, 0, 1, 1, 0, 0])
    assert timer.elapsed_seconds is not None
    assert result.resolved_inputs["scenario"] == (
        "detectiv.scenarios.reconstruction.ReconstructionScenario"
    )
    assert result.resolved_inputs["data"] == {
        "train": {
            "dataset_id": "images",
            "series_id": "series",
            "image_shape": [1, 4, 4],
            "window_count": 2,
            "series_length": 8,
        },
        "validation": None,
        "test": {
            "dataset_id": "images",
            "series_id": "series",
            "image_shape": [1, 4, 4],
            "window_count": 2,
            "series_length": 6,
        },
    }
    trainer = result.resolved_inputs["trainer"]
    scoring_plans = result.resolved_inputs["scoring_plans"]
    assert isinstance(trainer, Mapping)
    assert isinstance(scoring_plans, Sequence)
    assert isinstance(scoring_plans[0], Mapping)
    assert trainer["epochs"] == 1
    assert scoring_plans[0]["name"] == "mean_squared_window"
    performance = result.resolved_inputs["performance"]
    assert isinstance(performance, Mapping)
    assert set(performance) == {
        "training_seconds",
        "validation_seconds",
        "scoring_seconds",
        "propagation_seconds",
    }
    assert all(
        isinstance(duration, float) and duration >= 0
        for duration in performance.values()
    )
    json.dumps(result.resolved_inputs)


def test_completion_callback_failure_propagates() -> None:
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
        series_id="series",
        series_length=4,
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
        series_id="series",
        series_length=4,
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


def test_reconstruction_scenario_requires_aligned_score_references() -> None:
    references = (
        WindowReference("series", 0, 4, 4),
        WindowReference("series", 2, 6, 4),
    )
    images = _images(
        values=[np.zeros((1, 4, 4)), np.ones((1, 4, 4))],
        references=references,
        labels=np.array([False, True]),
        point_labels=np.array([0, 0, 1, 1, 0, 0]),
        series_length=6,
    )

    with pytest.raises(ValueError, match="preserve test window references"):
        _scenario(images, images, scorer=MisalignedScorer()).run()


def _images(
    *,
    values: list[np.ndarray],
    references: tuple[WindowReference, ...],
    labels: np.ndarray,
    series_length: int,
    point_labels: np.ndarray | None = None,
) -> ImageDataset:
    return ImageDataset(
        "images",
        image_shape=ImageShape(1, 4, 4),
        window_references=references,
        source=ArrayImageSource(values),
        window_labels=labels,
        series_id="series",
        series_length=series_length,
        point_labels=point_labels,
    )


def _scenario(
    train: ImageDataset,
    test: ImageDataset,
    *,
    callbacks: tuple[BaseCallback[ReconstructionReport], ...] = (),
    reproducibility: ReproducibilitySettings | None = None,
    scorer: ReconstructionScorer | None = None,
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
                scorer or MeanSquaredWindowReconstructionError(),
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
