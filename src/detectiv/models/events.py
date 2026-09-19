from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingEpochEvent:
    epoch: int
    training_loss: float
    validation_loss: float | None
    learning_rates: tuple[float, ...]
    elapsed_seconds: float
