import numpy as np

from detectiv.data import TimeSeries
from detectiv.datasets import TemporalBoundary, TemporalSplitter, TimeSeriesDataset


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
                "left": TemporalBoundary(train_end=2),
                "right": TemporalBoundary(train_end=3),
            }
        )
    )

    assert split.train.series_ids == ("left", "right")
    assert split.train["left"].n_timesteps == 2
    assert split.test["right"].n_timesteps == 5
    assert split.validation is None
