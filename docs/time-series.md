# Time Series

Detectiv represents each signal as a validated `TimeSeries`, then preserves its
temporal order through splitting, preprocessing, and windowing.

## Create And Split Data

Create a series with finite numeric values. One-dimensional input represents a
single feature; use named columns when feature identity matters across datasets.

```python
import numpy as np

from detectiv.time_series import (
    TemporalHoldout,
    TimeSeries,
)

series = TimeSeries(
    np.array([[0.2, 8.0], [0.4, 8.0], [0.7, 8.0], [1.1, 8.0]]),
    labels=np.array([0, 0, 0, 1]),
    feature_names=("temperature", "constant_sensor"),
    series_id="machine-1",
    metadata={"source": "example"},
)

rule = TemporalHoldout(test_start=3, validation_fraction=0.5)
splits = series.split(rule)
```

One scenario consumes one `TimeSeries`. Its immutable metadata and series ID are
preserved by temporal segments and preprocessing transformations.

## Fit Preprocessing

Fit preprocessing only on `splits.train`, then transform each partition. Fitted
preprocessors reject changed or reordered named feature schemas rather than
silently applying positional transformations.

```python
from detectiv.time_series.preprocessing import (
    ConstantFeatureRemoval,
    MinMaxScaling,
    PreprocessingPipeline,
)

preprocessing = PreprocessingPipeline((ConstantFeatureRemoval(), MinMaxScaling()))
preprocessing.fit(splits.train)

train = preprocessing.transform(splits.train)
test = preprocessing.transform(splits.test)
```

`ConstantFeatureRemoval` determines retained features from the pooled range of
all training timesteps. `MinMaxScaling` does not clip test values beyond the
training range.

## Extract Windows

Choose a window length and configure the tail explicitly. The default stride is
the length, producing contiguous non-overlapping windows.

```python
from detectiv.time_series.windowing import TailPolicy, WindowSpec

spec = WindowSpec(length=64, stride=16, tail=TailPolicy.EDGE_PAD)
windows = spec.windower().transform(train)

print(windows.values.shape)
print(windows.starts, windows.valid_lengths)
```

`WindowBatch.starts`, `stops`, and `valid_lengths` always refer to real source
observations. Padding affects values only, so `point_coverage()` never counts
padded positions. `WindowSpec.labeling` describes how to reduce labels when a
consumer needs one label per window; `Windower` itself returns values and
positions without consuming labels.

For exact parameters and contracts, see the [series API](api/time-series/series.md),
[preprocessing API](api/time-series/preprocessing/pipeline.md), and [windowing
API](api/time-series/windowing/core.md).
