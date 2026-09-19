from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from detectiv.callbacks.base import BaseCallback
from detectiv.images import ImageDataset
from detectiv.models.autoencoders import (
    Autoencoder,
    AutoencoderTrainer,
    TrainingHistory,
)
from detectiv.models.events import TrainingEpochEvent
from detectiv.scenarios.training import TrainingMode
from detectiv.scoring import (
    ReconstructionScoringPlan,
    WindowEvidenceBatch,
)
from detectiv.time_series import TemporalSplit


@dataclass(frozen=True)
class ReconstructionScenarioResult:
    window_scores: Mapping[str, WindowEvidenceBatch]
    point_scores: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]
    training: TrainingHistory
    callbacks: Mapping[str, BaseCallback] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def training_losses(self) -> tuple[float, ...]:
        return self.training.training_losses

    @property
    def validation_losses(self) -> tuple[float, ...]:
        return self.training.validation_losses


@dataclass(frozen=True)
class ReconstructionScenarioInspection:
    train_images: int
    validation_images: int | None
    test_images: int
    image_shape: tuple[int, int, int]
    scoring_plans: tuple[str, ...]
    callbacks: tuple[str, ...]

    def summary(self) -> str:
        validation = (
            "none" if self.validation_images is None else str(self.validation_images)
        )
        return "\n".join(
            (
                f"images: train={self.train_images}, validation={validation}, "
                f"test={self.test_images}, shape={self.image_shape}",
                f"scoring plans: {', '.join(self.scoring_plans)}",
                f"callbacks: {', '.join(self.callbacks) or 'none'}",
            )
        )


class ReconstructionScenario:
    def __init__(
        self,
        *,
        images: TemporalSplit[ImageDataset],
        model: Autoencoder,
        training_mode: TrainingMode,
        scoring_plans: Sequence[ReconstructionScoringPlan],
        callbacks: Sequence[BaseCallback] = (),
        trainer: AutoencoderTrainer | None = None,
    ) -> None:
        self.images = images
        self.model = model
        self.training_mode = training_mode
        if not scoring_plans:
            raise ValueError("at least one scoring plan is required")
        if len({plan.name for plan in scoring_plans}) != len(scoring_plans):
            raise ValueError("scoring plan names must be unique")
        if len({callback.name for callback in callbacks}) != len(callbacks):
            raise ValueError("callback names must be unique")
        self.scoring_plans = tuple(scoring_plans)
        self.callbacks = tuple(callbacks)
        self.trainer = trainer or AutoencoderTrainer()

    def run(self) -> ReconstructionScenarioResult:
        try:
            self._notify_started()
            if not len(self.images.test):
                raise ValueError("test images must contain at least one window")
            partition = self.training_mode.partition(
                self.images.train,
                self.images.validation,
            )
            training = self.trainer.fit(
                self.model,
                self.images.train,
                partition.training_indices,
                validation=partition.validation_images,
                validation_indices=partition.validation_indices,
                on_epoch_finished=self._notify_epoch_finished,
            )
            window_scores = {
                plan.name: plan.scorer.score(self.model, self.images.test)
                for plan in self.scoring_plans
            }
            self._validate_scores(window_scores)
            result = ReconstructionScenarioResult(
                window_scores=MappingProxyType(window_scores),
                point_scores=MappingProxyType(
                    {
                        plan.name: self._propagate(plan, window_scores[plan.name])
                        for plan in self.scoring_plans
                    }
                ),
                training=training,
                callbacks=MappingProxyType(
                    {callback.name: callback for callback in self.callbacks}
                ),
            )
        except BaseException as error:
            self._notify_failed(error)
            raise
        self._notify_finished(result)
        return result

    def inspect(self) -> ReconstructionScenarioInspection:
        return ReconstructionScenarioInspection(
            train_images=len(self.images.train),
            validation_images=None
            if self.images.validation is None
            else len(self.images.validation),
            test_images=len(self.images.test),
            image_shape=self.images.train.image_shape.shape,
            scoring_plans=tuple(plan.name for plan in self.scoring_plans),
            callbacks=tuple(callback.name for callback in self.callbacks),
        )

    def _propagate(
        self, plan: ReconstructionScoringPlan, scores: WindowEvidenceBatch
    ) -> Mapping[str, Mapping[str, np.ndarray]]:
        series_ids = dict.fromkeys(
            reference.series_id for reference in scores.references
        )
        point_scores = {
            point_scoring.name: MappingProxyType(
                {
                    series_id: _readonly(
                        point_scoring.aggregator.aggregate(
                            point_scoring.assignment.assign(
                                scores.for_series(series_id)
                            ),
                            self.images.test.series_lengths[series_id],
                        )
                    )
                    for series_id in series_ids
                }
            )
            for point_scoring in plan.point_scoring
        }
        return MappingProxyType(point_scores)

    def _validate_scores(self, scores: Mapping[str, WindowEvidenceBatch]) -> None:
        expected_series = set(self.images.test.series_lengths)
        for plan, values in scores.items():
            scored_series = {reference.series_id for reference in values.references}
            if missing := expected_series - scored_series:
                raise ValueError(
                    f"scoring plan {plan!r} produced no scores for test series: "
                    f"{', '.join(sorted(missing))}"
                )

    def _notify_started(self) -> None:
        for callback in self.callbacks:
            callback.on_run_started()

    def _notify_epoch_finished(self, event: TrainingEpochEvent) -> None:
        for callback in self.callbacks:
            callback.on_epoch_finished(event)

    def _notify_finished(self, result: ReconstructionScenarioResult) -> None:
        for callback in self.callbacks:
            callback.on_run_finished(result)

    def _notify_failed(self, error: BaseException) -> None:
        for callback in self.callbacks:
            callback.on_run_failed(error)


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result
