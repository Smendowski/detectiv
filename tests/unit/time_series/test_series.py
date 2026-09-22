import numpy as np
import pytest

from detectiv.time_series import TemporalBoundary, TemporalHoldout, TimeSeries


def test_univariate_series_uses_a_single_feature() -> None:
    values = np.array([1, 2, 3])
    labels = np.array([0, 1, 0])

    series = TimeSeries(
        values,
        labels=labels,
        feature_names=["signal"],
        series_id="example",
    )

    assert series.values.shape == (3, 1)
    assert series.values.dtype == np.float64
    assert series.labels is not None
    assert series.labels.tolist() == [False, True, False]
    assert series.n_timesteps == 3
    assert series.n_features == 1
    assert series.is_univariate
    assert series.series_id == "example"


def test_series_owns_read_only_data() -> None:
    values = np.array([[1.0], [2.0]])
    series = TimeSeries(values)

    values[0, 0] = 99.0

    assert series.values[0, 0] == 1.0
    assert not series.values.flags.writeable


def test_series_rejects_values_that_overflow_during_normalization() -> None:
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("longdouble has no wider range than float64 on this platform")
    values = np.array([np.longdouble(np.finfo(np.float64).max) * 2])

    with pytest.raises(ValueError, match="remain finite"):
        TimeSeries(values)


@pytest.mark.parametrize(
    ("values", "labels", "message"),
    [
        (np.array([1.0, np.nan]), None, "NaN or infinity"),
        (np.ones((2, 2, 2)), None, "shape"),
        (np.ones(3), np.array([0, 1]), "shape"),
        (np.ones(3), np.array([0, 2, 1]), "binary"),
    ],
)
def test_invalid_series_input_is_rejected(
    values: np.ndarray, labels: np.ndarray | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TimeSeries(values, labels=labels)


def test_unlabeled_series_is_supported() -> None:
    series = TimeSeries(np.ones((4, 2)))

    assert series.labels is None
    assert series.n_features == 2


def test_series_can_be_split_in_temporal_order() -> None:
    series = TimeSeries(
        np.arange(12),
        labels=np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        series_id="tao",
        metadata={"source": "test"},
    )

    split = series.split(TemporalBoundary(train_end=5, validation_end=8))

    assert split.train.values[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert split.validation is not None
    assert split.validation.labels is not None
    assert split.validation.labels.tolist() == [False, False, True]
    assert split.test.values[:, 0].tolist() == [8.0, 9.0, 10.0, 11.0]
    assert split.train.series_id == "tao"
    assert split.validation.series_id == "tao"
    assert split.test.series_id == "tao"
    assert split.train.metadata == {"source": "test"}
    with pytest.raises(TypeError):
        split.test.metadata["source"] = "other"  # type: ignore[index]


def test_series_can_be_split_without_validation() -> None:
    split = TimeSeries(np.arange(6)).split(TemporalBoundary(train_end=2))

    assert split.validation is None
    assert split.train.n_timesteps == 2
    assert split.test.n_timesteps == 4


@pytest.mark.parametrize("train_end", [6, 7])
def test_invalid_temporal_split_is_rejected(train_end: int) -> None:
    with pytest.raises(ValueError):
        TimeSeries(np.arange(6)).split(TemporalBoundary(train_end))


def test_series_split_resolves_a_temporal_holdout() -> None:
    split = TimeSeries(np.arange(10)).split(
        TemporalHoldout(test_start=8, validation_fraction=0.25)
    )

    assert split.train.values.ravel().tolist() == list(range(6))
    assert split.validation is not None
    assert split.validation.values.ravel().tolist() == [6, 7]
    assert split.test.values.ravel().tolist() == [8, 9]


@pytest.mark.parametrize("value", [2.5, True])
def test_temporal_boundaries_require_integral_positions(value: object) -> None:
    with pytest.raises(ValueError, match="integer"):
        TemporalBoundary(value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integer"):
        TemporalHoldout(value, 0.2)  # type: ignore[arg-type]
