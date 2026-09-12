import numpy as np

from detectiv.ts2i.channelization.base import Channelization
from detectiv.ts2i.channelization.context import TimestampContext


class MSMChannelization(Channelization):
    def __init__(self, context: TimestampContext) -> None:
        super().__init__(context)

    def transform(self, window: np.ndarray) -> tuple[np.ndarray, ...]:
        if window.ndim != 2:
            raise ValueError("MSM channelization requires a two-dimensional window")
        return (
            window.mean(axis=1),
            window.std(axis=1),
            window.max(axis=1),
        )
