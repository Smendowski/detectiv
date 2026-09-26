import numpy as np

from detectiv.images import ImageSize
from detectiv.time_series import TimeSeries
from detectiv.ts2i.channelization import (
    FeatureSelectionScope,
    HighestVariabilityFeatureChannelization,
)
from detectiv.ts2i.projection import (
    BaseProjectionStrategy,
    FixedProjectionStrategy,
    ProjectionScheme,
)
from detectiv.ts2i.transformations import Spiral


def test_fitted_projection_scheme_isolated_from_later_fits() -> None:
    first_train = TimeSeries(
        np.array([[0.0, 0.0], [0.5, 0.1], [1.0, 0.0]]),
        series_id="series",
    )
    second_train = TimeSeries(
        np.array([[0.0, 0.0], [1.0, 2.0], [0.0, 4.0]]),
        series_id="series",
    )
    strategy = FixedProjectionStrategy(
        ProjectionScheme(
            HighestVariabilityFeatureChannelization(1, FeatureSelectionScope.TRAINING)
        ).channels(Spiral())
    )

    assert isinstance(strategy, BaseProjectionStrategy)

    first = strategy.fit(first_train)
    size = ImageSize(height=4, width=4)
    image_before = first.render(first_train.values, size)
    second = strategy.fit(second_train)
    image_after = first.render(first_train.values, size)
    second_image = second.render(second_train.values, size)

    np.testing.assert_array_equal(image_after, image_before)
    assert not np.array_equal(second_image, image_before)
