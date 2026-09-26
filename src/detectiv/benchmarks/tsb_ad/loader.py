import csv
from pathlib import Path

import numpy as np

from detectiv.time_series import TimeSeries


def load_tsb_ad_csv(path: Path) -> TimeSeries:
    """Load one labeled TSB-AD CSV file into a time series.

    Args:
        path: CSV file whose final header column is named ``Label``.

    Returns:
        Time series with feature values, labels, feature names, and series ID.

    Raises:
        ValueError: If the file cannot be decoded or parsed, has no header,
            lacks a final ``Label`` column, or has rows with a different number
            of columns from its header.
    """
    with path.open(newline="", encoding="utf-8") as file:
        header = next(csv.reader(file), None)

    if header is None:
        raise ValueError("TSB-AD CSV must contain a header")

    if len(header) < 2 or header[-1] != "Label":
        raise ValueError("expected feature columns followed by a Label column")

    data = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
    if data.shape[1] != len(header):
        raise ValueError("CSV rows must match the header column count")

    return TimeSeries(
        data[:, :-1],
        labels=data[:, -1],
        feature_names=tuple(header[:-1]),
        series_id=path.stem,
    )
