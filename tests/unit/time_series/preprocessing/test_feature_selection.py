import numpy as np
import pytest

from detectiv.time_series import TimeSeries
from detectiv.time_series.preprocessing import (
    ConstantFeatureRemoval,
    PreprocessingPipeline,
)


def test_constant_feature_removal_uses_only_the_training_feature_mask() -> None:
    train = TimeSeries(np.array([[0.0, 1.0], [1.0, 1.0]]), series_id="series")
    test = TimeSeries(np.array([[2.0, 3.0]]), series_id="series")

    removal = ConstantFeatureRemoval().fit(train)

    assert removal.transform(train).values.tolist() == [[0.0], [1.0]]
    assert removal.transform(test).values.tolist() == [[2.0]]


def test_preprocessing_pipeline_applies_steps_in_order() -> None:
    dataset = TimeSeries(np.array([[0.0, 1.0], [2.0, 1.0]]), series_id="series")
    pipeline = PreprocessingPipeline((ConstantFeatureRemoval(),)).fit(dataset)

    assert pipeline.transform(dataset).n_features == 1


def test_failed_refit_preserves_the_previous_feature_selection() -> None:
    fitted = ConstantFeatureRemoval().fit(
        TimeSeries(np.array([[0.0, 1.0], [1.0, 1.0]]), series_id="series")
    )
    all_constant = TimeSeries(np.ones((2, 2)), series_id="series")

    with pytest.raises(ValueError, match="remove every feature"):
        fitted.fit(all_constant)

    assert fitted.transform(all_constant).n_features == 1


def test_constant_feature_removal_rejects_non_finite_tolerance() -> None:
    with pytest.raises(ValueError, match="finite"):
        ConstantFeatureRemoval(tolerance=float("nan"))


def test_constant_feature_removal_rejects_reordered_named_features() -> None:
    train = TimeSeries(
        np.array([[0.0, 1.0], [1.0, 1.0]]),
        feature_names=("kept", "constant"),
        series_id="series",
    )
    reordered = TimeSeries(
        np.array([[3.0, 2.0]]),
        feature_names=("constant", "kept"),
        series_id="series",
    )

    with pytest.raises(ValueError, match="feature names"):
        ConstantFeatureRemoval().fit(train).transform(reordered)
