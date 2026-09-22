import numpy as np

from detectiv.ts2i.channelization.base import Channelization


class MeanStdMaxChannelization(Channelization):
    """Summarize each time step with mean, standard deviation, and maximum."""

    def transform(self, window: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return three univariate feature-summary planes.

        Args:
            window: Two-dimensional time-major feature values.

        Returns:
            Per-time-step mean, standard deviation, and maximum values.

        Raises:
            ValueError: If the window is not two-dimensional.
        """
        if window.ndim != 2:
            raise ValueError("MSM channelization requires a two-dimensional window")
        return (
            window.mean(axis=1),
            window.std(axis=1),
            window.max(axis=1),
        )
