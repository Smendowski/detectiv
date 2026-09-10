import numpy as np
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler

from detectiv.data import TimeSeries
from detectiv.datasets import TimeSeriesDataset
from detectiv.preprocessing.base import TimeSeriesPreprocessor


class MinMaxScaling(TimeSeriesPreprocessor):
    def __init__(self) -> None:
        self._scaler: SklearnMinMaxScaler | None = None

    def fit(self, train: TimeSeriesDataset) -> "MinMaxScaling":
        values = tuple(train[series_id].values for series_id in train.series_ids)
        n_features = values[0].shape[1]
        if any(series.shape[1] != n_features for series in values):
            raise ValueError("all series must have the same number of features")
        self._scaler = SklearnMinMaxScaler().fit(np.concatenate(values))
        return self

    def transform(self, dataset: TimeSeriesDataset) -> TimeSeriesDataset:
        if self._scaler is None:
            raise RuntimeError("MinMaxScaling must be fitted before transforming data")
        return TimeSeriesDataset(
            dataset.dataset_id,
            {
                series_id: self._transform_series(dataset[series_id])
                for series_id in dataset.series_ids
            },
            metadata=dataset.metadata,
        )

    def _transform_series(self, series: TimeSeries) -> TimeSeries:
        if self._scaler is None:
            raise RuntimeError("MinMaxScaling must be fitted before transforming data")
        return TimeSeries(
            self._scaler.transform(series.values),
            labels=series.labels,
            feature_names=series.feature_names,
            sampling_rate=series.sampling_rate,
            series_id=series.series_id,
        )
