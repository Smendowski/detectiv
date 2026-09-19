from pathlib import Path

import numpy as np

from detectiv.callbacks import RunArtifactCallback
from detectiv.models.autoencoders import Autoencoder, AutoencoderTrainer
from detectiv.models.autoencoders.decoders import CNNDecoder
from detectiv.models.autoencoders.encoders import CNNEncoder
from detectiv.scenarios import (
    ReconstructionScenario,
    ReconstructionScenarioResult,
    SemiSupervisedTraining,
)
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
from detectiv.time_series.preprocessing import MinMaxScaling
from detectiv.time_series.windowing import TailPolicy, WindowSpec
from detectiv.ts2i import ImagePreparation
from detectiv.ts2i.channelization import IdentityChannelization
from detectiv.ts2i.projection import ConfiguredProjectionStrategy, ProjectionScheme
from detectiv.ts2i.transformations import LinePlot


def run(
    output_directory: Path = Path("artifacts/first_experiment"),
    *,
    visualize: bool = False,
) -> ReconstructionScenarioResult:
    values = np.sin(np.linspace(0, 4 * np.pi, 32))
    values[24:27] += 2.0
    labels = np.zeros(32, dtype=np.int32)
    labels[24:27] = 1
    dataset = TimeSeriesDataset(
        "synthetic",
        {"signal": TimeSeries(values, labels=labels, series_id="signal")},
    )
    preparation = (
        ImagePreparation(dataset)
        .split(TemporalSplitter({"signal": TemporalBoundary(16)}))
        .preprocess(MinMaxScaling())
        .window(
            train=WindowSpec(4, stride=4),
            test=WindowSpec(4, stride=1, tail=TailPolicy.EDGE_PAD),
        )
        .project(
            ConfiguredProjectionStrategy(
                ProjectionScheme(IdentityChannelization())
                .channels(LinePlot())
                .replicate(n_channels=3)
            )
        )
    )
    print(preparation.inspect((16, 16), seed=42).summary())
    images = preparation.build((16, 16), seed=42)
    reporter = RunArtifactCallback(
        output_directory,
        provenance={"dataset": "synthetic", "seed": 42},
        visualize=visualize,
    )
    scenario = ReconstructionScenario(
        images=images,
        model=Autoencoder(
            CNNEncoder(3, hidden_channels=(4,)),
            CNNDecoder(4, hidden_channels=(), output_channels=3),
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
        callbacks=(reporter,),
        trainer=AutoencoderTrainer(epochs=3, batch_size=2, shuffle_seed=42),
    )
    print(scenario.inspect().summary())
    result = scenario.run()
    if reporter.artifacts is not None:
        print(reporter.artifacts.manifest)
    return result


if __name__ == "__main__":
    run()
