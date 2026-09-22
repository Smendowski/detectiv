from abc import ABC, abstractmethod
from enum import StrEnum

import numpy as np

from detectiv.images import ImageSize


class TransformationInput(StrEnum):
    """Supported shape categories for a transformation input plane."""

    UNIVARIATE = "univariate"
    MULTIVARIATE = "multivariate"


class TS2ITransformation(ABC):
    """Render a channelization plane into a single image channel."""

    @property
    @abstractmethod
    def input_kinds(self) -> frozenset[TransformationInput]:
        """Return the input plane categories accepted by this transformation.

        Returns:
            Supported univariate and/or multivariate input categories.

        Raises:
            NotImplementedError: If a concrete transformation does not implement it.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(
        self,
        values: np.ndarray,
        size: ImageSize,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Render values at the requested height and width.

        Args:
            values: Input plane values accepted by ``input_kinds``.
            size: Output image height and width.
            rng: Optional random generator for stochastic transformations.

        Returns:
            A two-dimensional image plane.

        Raises:
            NotImplementedError: If a concrete transformation does not implement it.
        """
        raise NotImplementedError


def univariate_values(values: np.ndarray, transformation: str) -> np.ndarray:
    """Return a one-feature series or raise an input-shape error.

    Args:
        values: One-dimensional values or a two-dimensional one-feature plane.
        transformation: Transformation name included in validation errors.

    Returns:
        A one-dimensional time series.

    Raises:
        ValueError: If ``values`` has more than one feature.
    """
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    raise ValueError(f"{transformation} requires one feature")
