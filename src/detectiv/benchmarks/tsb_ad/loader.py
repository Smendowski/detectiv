from pathlib import Path

import numpy as np

from detectiv.time_series import TimeSeries


class TSBADCsvLoader:
    def load(self, path: Path) -> TimeSeries:
        try:
            header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
            values = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
        except (IndexError, OSError, UnicodeDecodeError, ValueError) as error:
            raise ValueError(f"failed to load TSB-AD CSV: {path}") from error
        if len(header) < 2 or header[-1] != "Label":
            raise ValueError("expected feature columns followed by a Label column")
        if values.shape[1] != len(header):
            raise ValueError("CSV rows must match the header column count")
        return TimeSeries(
            values[:, :-1],
            labels=values[:, -1],
            feature_names=tuple(header[:-1]),
            series_id=path.stem,
        )
