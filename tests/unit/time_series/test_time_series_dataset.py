import numpy as np
import pytest

from detectiv.time_series import (
    TemporalBoundary,
    TemporalHoldout,
    TemporalSplitter,
    TimeSeries,
    TimeSeriesDataset,
)


def test_dataset_is_split_per_series_in_temporal_order() -> None:
    dataset = TimeSeriesDataset(
        "example",
        {
            "left": TimeSeries(np.arange(6), series_id="left"),
            "right": TimeSeries(np.arange(8), series_id="right"),
        },
    )

    split = dataset.split(
        TemporalSplitter(
            {
                "right": TemporalBoundary(train_end=3),
                "left": TemporalBoundary(train_end=2),
            }
        )
    )

    assert split.train.series_ids == ("left", "right")
    assert split.train["left"].n_timesteps == 2
    assert split.test["right"].n_timesteps == 5
    assert split.validation is None


def test_temporal_holdout_keeps_validation_before_the_test_boundary() -> None:
    dataset = TimeSeriesDataset(
        "dataset",
        {"series": TimeSeries(np.arange(10), series_id="series")},
    )

    split = dataset.split(
        TemporalSplitter({"series": TemporalHoldout(8, validation_fraction=0.25)})
    )

    assert split.train["series"].values.ravel().tolist() == list(range(6))
    assert split.validation is not None
    assert split.validation["series"].values.ravel().tolist() == [6, 7]
    assert split.test["series"].values.ravel().tolist() == [8, 9]


def test_dataset_rejects_mismatched_feature_counts() -> None:
    with pytest.raises(ValueError, match="has 2 features; expected 1"):
        TimeSeriesDataset(
            "dataset",
            {
                "left": TimeSeries(np.ones((2, 1)), series_id="left"),
                "right": TimeSeries(np.ones((2, 2)), series_id="right"),
            },
        )


def test_dataset_rejects_mismatched_feature_name_order() -> None:
    with pytest.raises(ValueError, match="feature names must match"):
        TimeSeriesDataset(
            "dataset",
            {
                "left": TimeSeries(
                    np.ones((2, 2)),
                    feature_names=("temperature", "pressure"),
                    series_id="left",
                ),
                "right": TimeSeries(
                    np.ones((2, 2)),
                    feature_names=("pressure", "temperature"),
                    series_id="right",
                ),
            },
        )


def test_dataset_rejects_mixed_named_and_unnamed_features() -> None:
    with pytest.raises(ValueError, match="feature names must match"):
        TimeSeriesDataset(
            "dataset",
            {
                "named": TimeSeries(
                    np.ones((2, 1)), feature_names=("temperature",), series_id="named"
                ),
                "unnamed": TimeSeries(np.ones((2, 1)), series_id="unnamed"),
            },
        )
