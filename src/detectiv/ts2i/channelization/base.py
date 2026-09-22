from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from detectiv.time_series import TimeSeries


class Channelization(ABC):
    """Convert multivariate time windows into transformation input planes."""

    def fit(self, train: TimeSeries) -> Self:
        """Fit state from the training split and return this channelization.

        Args:
            train: Training-only data used to learn channelization state.

        Returns:
            This fitted channelization.
        """
        return self

    @abstractmethod
    def transform(self, window: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return the planes derived from one `(time, features)` window.

        Args:
            window: Time-major values for one source window.

        Returns:
            One or more transformation input planes.

        Raises:
            NotImplementedError: If a concrete channelization does not implement it.
        """
        raise NotImplementedError


class IdentityChannelization(Channelization):
    """Pass the complete input window through as one multivariate plane."""

    def transform(self, window: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return ``window`` unchanged as the sole plane.

        Args:
            window: Time-major values for one source window.

        Returns:
            A tuple containing the original window.
        """
        return (window,)
