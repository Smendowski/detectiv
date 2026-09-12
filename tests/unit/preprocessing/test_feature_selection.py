import numpy as np

from detectiv.data import TimeSeries
from detectiv.datasets import TimeSeriesDataset
from detectiv.preprocessing import ConstantFeatureRemoval, PreprocessingPipeline


def test_constant_feature_removal_uses_only_the_training_feature_mask() -> None:
    train = TimeSeriesDataset(
        "train",
        {
            "series": TimeSeries(
                np.array([[0.0, 1.0], [1.0, 1.0]]),
                series_id="series",
            )
        },
    )
    test = TimeSeriesDataset(
        "test",
        {"series": TimeSeries(np.array([[2.0, 3.0]]), series_id="series")},
    )

    removal = ConstantFeatureRemoval().fit(train)

    assert removal.transform(train)["series"].values.tolist() == [[0.0], [1.0]]
    assert removal.transform(test)["series"].values.tolist() == [[2.0]]


def test_preprocessing_pipeline_applies_steps_in_order() -> None:
    dataset = TimeSeriesDataset(
        "train",
        {
            "series": TimeSeries(
                np.array([[0.0, 1.0], [2.0, 1.0]]),
                series_id="series",
            )
        },
    )
    pipeline = PreprocessingPipeline((ConstantFeatureRemoval(),)).fit(dataset)

    assert pipeline.transform(dataset)["series"].n_features == 1
