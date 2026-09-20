from pathlib import Path

import numpy as np

from detectiv.images.io.readers import ImageFolderReader
from detectiv.models.autoencoders import Autoencoder, AutoencoderTrainer
from detectiv.models.autoencoders.decoders import CNNDecoder
from detectiv.models.autoencoders.encoders import CNNEncoder
from detectiv.scoring import MeanSquaredWindowReconstructionError
from detectiv.time_series import (
    TemporalBoundary,
    TemporalSplitter,
    TimeSeries,
    TimeSeriesDataset,
)
from detectiv.time_series.windowing import WindowSpec
from detectiv.ts2i import DataLoaderSettings, ImagePreparation, MaterializationSettings
from detectiv.ts2i.channelization import MeanStdMaxChannelization
from detectiv.ts2i.projection import FixedProjectionStrategy, ProjectionScheme
from detectiv.ts2i.transformations import Spiral


def test_materialized_ts2i_images_reload_as_trainable_image_folder(
    tmp_path: Path,
) -> None:
    dataset = TimeSeriesDataset(
        "telemetry",
        {
            "pump": TimeSeries(
                np.arange(48, dtype=np.float32).reshape(24, 2),
                labels=np.array([False] * 16 + [True] * 8),
                series_id="pump",
            )
        },
        metadata={"source": "integration"},
    )
    preparation = (
        ImagePreparation(dataset)
        .split(
            TemporalSplitter({"pump": TemporalBoundary(train_end=8, validation_end=16)})
        )
        .window(train=WindowSpec(2), validation=WindowSpec(2), test=WindowSpec(2))
        .project(
            FixedProjectionStrategy(
                ProjectionScheme(MeanStdMaxChannelization()).channels(
                    Spiral(), Spiral(), Spiral()
                )
            )
        )
    )
    materialized, _ = preparation.materialize(
        (4, 4),
        MaterializationSettings(tmp_path / "images", workers=2),
        seed=7,
    )

    restored = ImageFolderReader(tmp_path / "images").read()

    for expected, actual in zip(
        (materialized.train, materialized.validation, materialized.test),
        (restored.train, restored.validation, restored.test),
        strict=True,
    ):
        assert expected is not None
        assert actual is not None
        assert expected.metadata == actual.metadata
        assert expected.window_references == actual.window_references
        np.testing.assert_array_equal(expected.window_labels, actual.window_labels)
        for index in range(len(expected)):
            np.testing.assert_array_equal(expected[index], actual[index])

    loader = DataLoaderSettings(workers=0, pin_memory=True)
    model = Autoencoder(
        CNNEncoder(3, hidden_channels=(4,)),
        CNNDecoder(4, hidden_channels=(), output_channels=3),
    )
    AutoencoderTrainer(
        epochs=1,
        batch_size=2,
        device="cpu",
        shuffle_seed=7,
        data_loader=loader,
    ).fit(model, restored.train, range(len(restored.train)))
    scorer = MeanSquaredWindowReconstructionError(
        batch_size=2, device="cpu", data_loader=loader
    )
    expected_scores = scorer.score(model, materialized.test)
    restored_scores = scorer.score(model, restored.test)

    assert expected_scores.references == restored_scores.references
    np.testing.assert_allclose(expected_scores.values, restored_scores.values)
