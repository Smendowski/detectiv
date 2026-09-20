import csv
from pathlib import Path

import numpy as np

from detectiv.time_series import TimeSeries


class TSBADCsvLoader:
    """Load one labeled TSB-AD CSV file into a time series."""

    def load(self, path: Path) -> TimeSeries:
        """Load labeled feature values from a TSB-AD CSV file.

        Args:
            path: CSV file whose final header column is named ``Label``.

        Returns:
            Time series with feature values, labels, feature names, and series ID.

        Raises:
            ValueError: If the file cannot be decoded or parsed, has no header,
                lacks a final ``Label`` column, or has rows with a different
                number of columns from its header.
        """
        try:
            with path.open(newline="", encoding="utf-8") as file:
                header = next(csv.reader(file))
            values = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
        except (
            StopIteration,
            csv.Error,
            OSError,
            UnicodeDecodeError,
            ValueError,
        ) as error:
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
