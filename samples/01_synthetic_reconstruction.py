from pathlib import Path
from shutil import rmtree

import numpy as np

from detectiv.callbacks import TimingCallback
from detectiv.models.autoencoders import Autoencoder, AutoencoderTrainer
from detectiv.models.autoencoders.decoders import CNNDecoder
from detectiv.models.autoencoders.encoders import CNNEncoder
from detectiv.protocols import SemiSupervisedTraining
from detectiv.runs import configure_reproducibility
from detectiv.scenarios import ReconstructionScenario
from detectiv.scoring import (
    MeanPointScoreAggregator,
    MeanSquaredWindowReconstructionError,
    PointScoringPlan,
    ReconstructionScoringPlan,
    UniformPointAssignment,
)
from detectiv.time_series import (
    TemporalBoundary,
    TemporalSplitter,
    TimeSeries,
    TimeSeriesDataset,
)
from detectiv.time_series.windowing import WindowSpec
from detectiv.ts2i import ImagePreparation, MaterializationSettings
from detectiv.ts2i.channelization import IdentityChannelization
from detectiv.ts2i.projection import FixedProjectionStrategy, ProjectionScheme
from detectiv.ts2i.transformations import StateGrid

SEED = 7
TRAIN_END = 240
WINDOW_SIZE = 32
TRAIN_STRIDE = 16
TEST_STRIDE = 8
IMAGE_SIZE = (32, 32)
N_CHANNELS = 3
BATCH_SIZE = 16
EPOCHS = 20
OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "outputs" / Path(__file__).stem


def main() -> None:
    output = OUTPUT_DIRECTORY

    rmtree(output, ignore_errors=True)
    reproducibility = configure_reproducibility(seed=SEED)

    time = np.arange(384, dtype=np.float32)
    values = np.sin(2 * np.pi * time / 32) + 0.1 * np.sin(2 * np.pi * time / 9)
    labels = np.zeros_like(time, dtype=bool)
    labels[300:324] = True
    values[labels] += 2.0

    series_id = "synthetic"
    series = TimeSeries(values, labels=labels, series_id=series_id)
    dataset = TimeSeriesDataset(
        "synthetic",
        {series_id: series},
    )
    images = (
        ImagePreparation(dataset)
        .split(TemporalSplitter({series_id: TemporalBoundary(train_end=TRAIN_END)}))
        .window(
            train=WindowSpec(WINDOW_SIZE, stride=TRAIN_STRIDE),
            test=WindowSpec(WINDOW_SIZE, stride=TEST_STRIDE),
        )
        .project(
            FixedProjectionStrategy(
                ProjectionScheme(IdentityChannelization())
                .channels(StateGrid())
                .replicate(n_channels=N_CHANNELS)
            )
        )
        .materialize(
            IMAGE_SIZE,
            MaterializationSettings(output),
            reproducibility=reproducibility,
        )
    )

    timer = TimingCallback()
    scenario = ReconstructionScenario(
        images=images,
        model=Autoencoder(
            CNNEncoder(N_CHANNELS, hidden_channels=(8, 16)),
            CNNDecoder(16, hidden_channels=(8,), output_channels=N_CHANNELS),
        ),
        trainer=AutoencoderTrainer(epochs=EPOCHS, batch_size=BATCH_SIZE),
        training_mode=SemiSupervisedTraining(),
        scoring_plans=(
            ReconstructionScoringPlan(
                MeanSquaredWindowReconstructionError(batch_size=BATCH_SIZE),
                (
                    PointScoringPlan(
                        UniformPointAssignment(),
                        MeanPointScoreAggregator(),
                    ),
                ),
            ),
        ),
        callbacks=(timer,),
        reproducibility=reproducibility,
    )

    report = scenario.run()
    artifacts = report.write(output / "run")

    print(report.summary())
    print(f"Elapsed scenario time: {timer.elapsed_seconds:.3f} s")
    print(f"Run JSON: {artifacts.manifest}")
    print(f"Point-score archive: {artifacts.point_scores}")


if __name__ == "__main__":
    main()
