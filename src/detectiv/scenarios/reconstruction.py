from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType

import numpy as np

from detectiv.callbacks.base import BaseCallback
from detectiv.images import ImageDataset
from detectiv.models.autoencoders import Autoencoder, AutoencoderTrainer
from detectiv.protocols import TrainingMode
from detectiv.runs import JSONValue, ReproducibilitySettings
from detectiv.scenarios.base import BaseScenario
from detectiv.scenarios.results import ReconstructionScenarioResult
from detectiv.scoring import (
    ReconstructionScoringPlan,
    WindowEvidenceBatch,
)
from detectiv.time_series import TemporalSplit


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


class ReconstructionScenario(BaseScenario[ReconstructionScenarioResult]):
    scenario_type = "reconstruction"

    def __init__(
        self,
        *,
        images: TemporalSplit[ImageDataset],
        model: Autoencoder,
        training_mode: TrainingMode,
        scoring_plans: Sequence[ReconstructionScoringPlan],
        callbacks: Sequence[BaseCallback[ReconstructionScenarioResult]] = (),
        trainer: AutoencoderTrainer | None = None,
        reproducibility: ReproducibilitySettings | None = None,
    ) -> None:
        self.images = images
        self.model = model
        self.training_mode = training_mode
        if not scoring_plans:
            raise ValueError("at least one scoring plan is required")
        if len({plan.name for plan in scoring_plans}) != len(scoring_plans):
            raise ValueError("scoring plan names must be unique")
        self.scoring_plans = tuple(scoring_plans)
        super().__init__(callbacks=callbacks, reproducibility=reproducibility)
        self.trainer = trainer or AutoencoderTrainer()

    def _run(self) -> ReconstructionScenarioResult:
        if not len(self.images.test):
            raise ValueError("test images must contain at least one window")

        partition = self.training_mode.partition(
            self.images.train,
            self.images.validation,
        )
        training_started = perf_counter()
        training = self.trainer.fit(
            self.model,
            self.images.train,
            partition.training_indices,
            validation=partition.validation_images,
            validation_indices=partition.validation_indices,
            on_epoch_finished=self._notify_epoch_finished,
        )
        training_seconds = perf_counter() - training_started
        scoring_started = perf_counter()
        window_scores = {
            plan.name: plan.scorer.score(self.model, self.images.test)
            for plan in self.scoring_plans
        }
        self._validate_scores(window_scores)
        scoring_seconds = perf_counter() - scoring_started
        propagation_started = perf_counter()
        point_scores = MappingProxyType(
            {
                plan.name: self._propagate(plan, window_scores[plan.name])
                for plan in self.scoring_plans
            }
        )
        propagation_seconds = perf_counter() - propagation_started

        return ReconstructionScenarioResult(
            window_scores=MappingProxyType(window_scores),
            point_scores=point_scores,
            training=training,
            callbacks=MappingProxyType(
                {callback.name: callback for callback in self.callbacks}
            ),
            reproducibility=(
                MappingProxyType({})
                if self.reproducibility is None
                else self.reproducibility.record(device=training.device)
            ),
            resolved_inputs={
                "scenario": _type_name(self),
                "data": {
                    "train": _dataset_record(self.images.train),
                    "validation": (
                        None
                        if self.images.validation is None
                        else _dataset_record(self.images.validation)
                    ),
                    "test": _dataset_record(self.images.test),
                },
                "model": {
                    "type": _type_name(self.model),
                    "encoder": _component_record(self.model.encoder),
                    "bottleneck": (
                        None
                        if self.model.bottleneck is None
                        else _component_record(self.model.bottleneck)
                    ),
                    "decoder": _component_record(self.model.decoder),
                },
                "training_mode": {
                    **_component_record(self.training_mode),
                    "validation_holdout": (
                        None
                        if self.training_mode.validation_holdout is None
                        else _component_record(self.training_mode.validation_holdout)
                    ),
                },
                "trainer": _trainer_record(self.trainer),
                "scoring_plans": [
                    _scoring_plan_record(plan) for plan in self.scoring_plans
                ],
                "performance": {
                    "data_wait_seconds": training.data_wait_seconds,
                    "transfer_seconds": training.transfer_seconds,
                    "training_seconds": training_seconds,
                    "validation_seconds": training.validation_seconds,
                    "scoring_seconds": scoring_seconds,
                    "propagation_seconds": propagation_seconds,
                },
            },
        )

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


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


def _dataset_record(dataset: ImageDataset) -> dict[str, JSONValue]:
    record: dict[str, JSONValue] = {
        "dataset_id": dataset.dataset_id,
        "image_shape": list(dataset.image_shape.shape),
        "window_count": len(dataset),
        "series_lengths": dict(dataset.series_lengths),
    }
    if "ts2i_performance" in dataset.metadata:
        record["ts2i_performance"] = dataset.metadata["ts2i_performance"]  # type: ignore[assignment]
    return record


def _trainer_record(trainer: AutoencoderTrainer) -> dict[str, JSONValue]:
    return {
        "epochs": trainer.epochs,
        "batch_size": trainer.batch_size,
        "learning_rate": trainer.learning_rate,
        "weight_decay": trainer.weight_decay,
        "device": trainer.device,
        "shuffle_seed": trainer.shuffle_seed,
        "optimizer": _callable_name(trainer.optimizer),
        "scheduler_factory": (
            None
            if trainer.scheduler_factory is None
            else _callable_name(trainer.scheduler_factory)
        ),
        "transfer_strategy": _component_record(trainer.transfer_strategy),
        "loss": _component_record(trainer.loss),
        "early_stopping_patience": trainer.early_stopping_patience,
        "min_delta": trainer.min_delta,
        "restore_best_validation": trainer.restore_best_validation,
        "data_loader": {
            "workers": trainer.data_loader.workers,
            "prefetch_factor": trainer.data_loader.prefetch_factor,
            "persistent_workers": trainer.data_loader.persistent_workers,
            "pin_memory": trainer.data_loader.pin_memory,
        },
    }


def _scoring_plan_record(plan: ReconstructionScoringPlan) -> dict[str, JSONValue]:
    return {
        "name": plan.name,
        "scorer": _component_record(plan.scorer),
        "point_scoring": [
            {
                "name": point_plan.name,
                "assignment": _component_record(point_plan.assignment),
                "aggregator": _component_record(point_plan.aggregator),
            }
            for point_plan in plan.point_scoring
        ],
    }


def _component_record(component: object) -> dict[str, JSONValue]:
    settings = {
        name: value
        for name, value in vars(component).items()
        if isinstance(value, bool | int | float | str) or value is None
    }
    return {
        "type": _type_name(component),
        "settings": settings,
        "configuration": str(component),
    }


def _type_name(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _callable_name(value: object) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{name}"
