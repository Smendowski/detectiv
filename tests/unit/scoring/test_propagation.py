import numpy as np
import pytest

from detectiv.data import WindowReference
from detectiv.scoring import (
    MaxPropagationStrategy,
    MeanPropagationStrategy,
    MedianPropagationStrategy,
    SaliencyWeightedPropagationStrategy,
    TemporalColumnPropagationStrategy,
    UncoveredPolicy,
    WindowPointScoreBatch,
    WindowSaliencyBatch,
    WindowScoreBatch,
)


def test_overlapping_window_scores_are_aggregated_without_labels() -> None:
    scores = WindowScoreBatch(
        np.array([1.0, 3.0]),
        (
            WindowReference("series", 0, 3, 3),
            WindowReference("series", 1, 4, 3),
        ),
    )

    mean = MeanPropagationStrategy().transform(scores, 4)
    maximum = MaxPropagationStrategy().transform(scores, 4)
    median = MedianPropagationStrategy().transform(scores, 4)

    np.testing.assert_array_equal(mean, [1.0, 2.0, 2.0, 3.0])
    np.testing.assert_array_equal(maximum, [1.0, 3.0, 3.0, 3.0])
    np.testing.assert_array_equal(median, [1.0, 2.0, 2.0, 3.0])


def test_tail_padding_respects_the_reference_valid_length() -> None:
    scores = WindowScoreBatch(
        np.array([1.0, 3.0]),
        (
            WindowReference("series", 0, 4, 4),
            WindowReference("series", 4, 8, 2),
        ),
    )

    np.testing.assert_array_equal(
        MeanPropagationStrategy().transform(scores, 6), [1, 1, 1, 1, 3, 3]
    )


def test_pointwise_window_scores_only_aggregate_overlaps() -> None:
    scores = WindowPointScoreBatch(
        (np.array([1.0, 2.0, 3.0]), np.array([5.0, 6.0, 7.0])),
        (
            WindowReference("series", 0, 3, 3),
            WindowReference("series", 1, 4, 3),
        ),
    )

    np.testing.assert_array_equal(
        MeanPropagationStrategy().transform(scores, 4), [1.0, 3.5, 4.5, 7.0]
    )
    np.testing.assert_array_equal(
        TemporalColumnPropagationStrategy().transform(scores, 4), [1.0, 3.5, 4.5, 7.0]
    )


def test_temporal_column_propagation_requires_point_scores() -> None:
    scores = WindowScoreBatch(
        np.array([1.0]),
        (WindowReference("series", 0, 2, 2),),
    )

    with pytest.raises(TypeError, match="time-resolved"):
        TemporalColumnPropagationStrategy().transform(scores, 2)


def test_saliency_weighted_propagation_preserves_each_window_score() -> None:
    scores = WindowSaliencyBatch(
        np.array([2.0, 4.0]),
        (np.array([1.0, 3.0]), np.array([2.0, 2.0])),
        (
            WindowReference("series", 0, 2, 2),
            WindowReference("series", 1, 3, 2),
        ),
    )

    result = SaliencyWeightedPropagationStrategy().transform(scores, 3)

    np.testing.assert_array_equal(result, [1.0, 3.5, 4.0])


def test_only_trailing_gaps_can_use_edge_padding() -> None:
    scores = WindowScoreBatch(
        np.array([1.0]),
        (WindowReference("series", 0, 2, 2),),
    )

    with pytest.raises(ValueError, match="uncovered"):
        MeanPropagationStrategy().transform(scores, 3)
    np.testing.assert_array_equal(
        MeanPropagationStrategy(UncoveredPolicy.EDGE_PAD).transform(scores, 3),
        [1, 1, 1],
    )
