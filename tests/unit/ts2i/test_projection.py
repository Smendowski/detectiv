import numpy as np

from detectiv.time_series import TimeSeries, TimeSeriesDataset
from detectiv.ts2i.channelization import (
    FeatureSelectionScope,
    HighestVariabilityFeatureChannelization,
)
from detectiv.ts2i.projection import ConfiguredProjectionStrategy, ProjectionScheme
from detectiv.ts2i.transformations import Spiral


def test_fitted_projection_scheme_isolated_from_later_fits() -> None:
    first_train = TimeSeriesDataset(
        "first",
        {
            "series": TimeSeries(
                np.array([[0.0, 0.0], [0.5, 0.1], [1.0, 0.0]]),
                series_id="series",
            )
        },
    )
    second_train = TimeSeriesDataset(
        "second",
        {
            "series": TimeSeries(
                np.array([[0.0, 0.0], [1.0, 2.0], [0.0, 4.0]]),
                series_id="series",
            )
        },
    )
    strategy = ConfiguredProjectionStrategy(
        ProjectionScheme(
            HighestVariabilityFeatureChannelization(1, FeatureSelectionScope.TRAINING)
        ).channels(Spiral())
    )

    first = strategy.fit(first_train)
    image_before = first.render(first_train["series"].values, (4, 4))
    second = strategy.fit(second_train)
    image_after = first.render(first_train["series"].values, (4, 4))
    second_image = second.render(second_train["series"].values, (4, 4))

    np.testing.assert_array_equal(image_after, image_before)
    assert not np.array_equal(second_image, image_before)
