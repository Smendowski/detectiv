import numpy as np
import pytest

from detectiv.scoring import (
    DirectPointAssignment,
    MaxPointScoreAggregator,
    MeanPointScoreAggregator,
    MedianPointScoreAggregator,
    SaliencyWeightedPointAssignment,
    UncoveredPolicy,
    UniformPointAssignment,
    WindowPointScoreBatch,
    WindowSaliencyBatch,
    WindowScoreBatch,
)
from detectiv.time_series.windowing import WindowReference


def test_uniform_assignment_and_aggregation_resolve_overlaps() -> None:
    scores = WindowScoreBatch(
        np.array([1.0, 3.0]),
        (
            WindowReference("series", 0, 3, 3),
            WindowReference("series", 1, 4, 3),
        ),
    )

    contributions = UniformPointAssignment().assign(scores)

    np.testing.assert_array_equal(contributions.values[0], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(contributions.values[1], [3.0, 3.0, 3.0])
    np.testing.assert_array_equal(
        MeanPointScoreAggregator().aggregate(contributions, 4), [1.0, 2.0, 2.0, 3.0]
    )
    np.testing.assert_array_equal(
        MaxPointScoreAggregator().aggregate(contributions, 4), [1.0, 3.0, 3.0, 3.0]
    )
    np.testing.assert_array_equal(
        MedianPointScoreAggregator().aggregate(contributions, 4), [1.0, 2.0, 2.0, 3.0]
    )


def test_uniform_assignment_respects_the_reference_valid_length() -> None:
    scores = WindowScoreBatch(
        np.array([1.0, 3.0]),
        (
            WindowReference("series", 0, 4, 4),
            WindowReference("series", 4, 8, 2),
        ),
    )

    np.testing.assert_array_equal(
        MeanPointScoreAggregator().aggregate(
            UniformPointAssignment().assign(scores), 6
        ),
        [1, 1, 1, 1, 3, 3],
    )


def test_direct_assignment_only_requires_overlap_aggregation() -> None:
    scores = WindowPointScoreBatch(
        (np.array([1.0, 2.0, 3.0]), np.array([5.0, 6.0, 7.0])),
        (
            WindowReference("series", 0, 3, 3),
            WindowReference("series", 1, 4, 3),
        ),
    )

    contributions = DirectPointAssignment().assign(scores)

    assert contributions is scores
    np.testing.assert_array_equal(
        MeanPointScoreAggregator().aggregate(contributions, 4), [1.0, 3.5, 4.5, 7.0]
    )
    np.testing.assert_array_equal(
        MedianPointScoreAggregator().aggregate(contributions, 4),
        [1.0, 3.5, 4.5, 7.0],
    )


def test_direct_assignment_requires_point_scores() -> None:
    scores = WindowScoreBatch(
        np.array([1.0]),
        (WindowReference("series", 0, 2, 2),),
    )

    with pytest.raises(TypeError, match="one score per window point"):
        DirectPointAssignment().assign(scores)


def test_saliency_assignment_preserves_each_window_score_on_average() -> None:
    scores = WindowSaliencyBatch(
        np.array([2.0, 4.0]),
        (np.array([1.0, 3.0]), np.array([2.0, 2.0])),
        (
            WindowReference("series", 0, 2, 2),
            WindowReference("series", 1, 3, 2),
        ),
    )

    contributions = SaliencyWeightedPointAssignment().assign(scores)

    np.testing.assert_array_equal(contributions.values[0], [1.0, 3.0])
    np.testing.assert_array_equal(contributions.values[1], [4.0, 4.0])
    np.testing.assert_array_equal(
        MeanPointScoreAggregator().aggregate(contributions, 3), [1.0, 3.5, 4.0]
    )


def test_aggregator_resolves_uncovered_points() -> None:
    scores = WindowScoreBatch(
        np.array([1.0]),
        (WindowReference("series", 0, 2, 2),),
    )
    contributions = UniformPointAssignment().assign(scores)

    with pytest.raises(ValueError, match="uncovered"):
        MeanPointScoreAggregator().aggregate(contributions, 3)
    np.testing.assert_array_equal(
        MeanPointScoreAggregator(UncoveredPolicy.EDGE_PAD).aggregate(contributions, 3),
        [1, 1, 1],
    )
