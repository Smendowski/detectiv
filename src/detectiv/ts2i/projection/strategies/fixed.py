from copy import deepcopy
from dataclasses import replace

from detectiv.time_series import TimeSeries
from detectiv.ts2i.projection.schemes import ProjectionScheme
from detectiv.ts2i.projection.strategies.base import BaseProjectionStrategy


class FixedProjectionStrategy(BaseProjectionStrategy):
    """Fit a configured projection scheme without changing its composition."""

    def __init__(self, scheme: ProjectionScheme) -> None:
        """Store the channelization and transformation scheme to fit.

        Args:
            scheme: Configured scheme whose channelization may require fitting.
        """
        self.scheme = scheme

    def fit(self, train: TimeSeries) -> ProjectionScheme:
        """Clone and fit the scheme channelization on training data.

        Args:
            train: Training-only data used by the channelization.

        Returns:
            A scheme with independent fitted channelization state.
        """
        channelization = deepcopy(self.scheme.channelization)
        channelization.fit(train)
        return replace(self.scheme, channelization=channelization)
