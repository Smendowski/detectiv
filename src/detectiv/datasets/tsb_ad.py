from pathlib import Path

import numpy as np

from detectiv.data import TimeSeries


class TSBADCsvLoader:
    def load(self, path: Path) -> TimeSeries:
        values = np.loadtxt(path, delimiter=",", skiprows=1)
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError("expected feature columns followed by a Label column")
        return TimeSeries(
            values[:, :-1],
            labels=values[:, -1],
            series_id=path.stem,
        )
