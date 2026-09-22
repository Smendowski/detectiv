from pathlib import Path

import numpy as np
import pytest
import torch

from detectiv.models.autoencoders import Autoencoder
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


def test_chunked_process_materialization_matches_synchronous(tmp_path: Path) -> None:
    preparation = _preparation()
    synchronous = preparation.materialize(
        (4, 4), MaterializationSettings(tmp_path / "synchronous", workers=0)
    )
    concurrent = preparation.materialize(
        (4, 4), MaterializationSettings(tmp_path / "concurrent", workers=2)
    )

    for expected, actual in zip(
        (synchronous.train, synchronous.test),
        (concurrent.train, concurrent.test),
        strict=True,
    ):
        assert expected.window_references == actual.window_references
        assert expected.window_labels is not None
        assert actual.window_labels is not None
        assert np.array_equal(expected.window_labels, actual.window_labels)
        assert expected.point_labels is not None
        assert actual.point_labels is not None
        np.testing.assert_array_equal(
            expected.point_labels["series"], actual.point_labels["series"]
        )
        assert all(
            np.array_equal(expected[index], actual[index])
            for index in range(len(expected))
        )
    assert synchronous.materialization.selected_workers == 0
    assert concurrent.materialization.selected_workers == 2
    assert concurrent.provenance["source"] == "materialized"
    assert concurrent.provenance["location"] == str(tmp_path / "concurrent")
    assert (tmp_path / "concurrent" / "materialization.json").is_file()

    scorer = MeanSquaredWindowReconstructionError(batch_size=2, device="cpu")
    model = Autoencoder(
        CNNEncoder(3, hidden_channels=(4,)),
        CNNDecoder(4, hidden_channels=(), output_channels=3),
    )
    expected_scores = scorer.score(model, synchronous.test)
    actual_scores = scorer.score(model, concurrent.test)
    assert expected_scores.references == actual_scores.references
    np.testing.assert_allclose(expected_scores.values, actual_scores.values)


def test_auto_falls_back_when_no_candidate_meets_threshold(tmp_path: Path) -> None:
    images = _preparation().materialize(
        (4, 4),
        MaterializationSettings(
            tmp_path / "auto", workers="auto", improvement_threshold=2.0
        ),
    )

    assert images.materialization.selected_workers == 0
    assert images.materialization.fallback_reason is not None
    assert images.materialization.candidates[0].workers == 0


@pytest.mark.parametrize("workers", [-1, "thread"])
def test_materialization_rejects_invalid_worker_modes(
    tmp_path: Path, workers: object
) -> None:
    with pytest.raises(ValueError, match="workers"):
        MaterializationSettings(tmp_path / "images", workers=workers)  # type: ignore[arg-type]


def test_max_is_bounded_by_independent_windows(tmp_path: Path) -> None:
    images = _preparation().materialize(
        (4, 4), MaterializationSettings(tmp_path / "max", workers="max")
    )

    assert images.materialization.selected_workers <= len(images.train) + len(
        images.test
    )


def test_materialization_removes_staging_directory_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from detectiv.ts2i import materialization

    def fail(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(materialization, "_write_range", fail)
    destination = tmp_path / "interrupted"

    with pytest.raises(KeyboardInterrupt):
        _preparation().materialize(
            (4, 4), MaterializationSettings(destination, workers=0)
        )

    assert not destination.exists()


def test_materialization_rejects_an_incomplete_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from detectiv.ts2i import materialization

    monkeypatch.setattr(materialization, "_write_range", lambda *args: None)
    destination = tmp_path / "incomplete"

    with pytest.raises(RuntimeError, match="did not write expected image"):
        _preparation().materialize(
            (4, 4), MaterializationSettings(destination, workers=0)
        )

    assert not destination.exists()


def test_explicit_workers_cannot_exceed_independent_windows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="independent windows"):
        _preparation().materialize(
            (4, 4), MaterializationSettings(tmp_path / "images", workers=99)
        )


def test_loader_settings_are_safe_on_cpu() -> None:
    settings = DataLoaderSettings(workers=0, pin_memory=True)

    assert settings.pin_memory
    with pytest.raises(ValueError, match="persistent"):
        DataLoaderSettings(persistent_workers=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_loader_settings_support_cuda_pinning() -> None:
    settings = DataLoaderSettings(workers=1, pin_memory=True)

    assert settings.pin_memory


def _preparation() -> ImagePreparation:
    dataset = TimeSeriesDataset(
        "series",
        {
            "series": TimeSeries(
                np.arange(16, dtype=float),
                labels=np.array([False] * 8 + [True] * 8),
                series_id="series",
            )
        },
    )
    projection = FixedProjectionStrategy(
        ProjectionScheme(MeanStdMaxChannelization()).channels(
            Spiral(), Spiral(), Spiral()
        )
    )
    return (
        ImagePreparation(dataset)
        .split(TemporalSplitter({"series": TemporalBoundary(train_end=8)}))
        .window(train=WindowSpec(2), test=WindowSpec(2))
        .project(projection)
    )
