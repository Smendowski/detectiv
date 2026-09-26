# TSB-AD

[TSB-AD](https://github.com/TheDatumOrg/TSB-AD) is a time-series anomaly
detection benchmark introduced in Liu and Paparrizos, ["The Elephant in the
Room: Towards A Reliable Time-Series Anomaly Detection
Benchmark"](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html),
NeurIPS 2024 Datasets and Benchmarks Track.

Detectiv uses the upstream TSB-AD package only for its ACF window estimator and
benchmark metrics. It loads TSB-AD CSV datasets directly from the local
`data/` directory.

## What Detectiv Provides

The integration has three small entry points:

- `load_tsb_ad_csv()` reads a single CSV file.
- `TSBADCollectionLoader` reads the boundary from each filename and returns a
  ready-to-use `TimeSeriesSplit`.
- `TSBADAdapter` provides the two upstream operations Detectiv needs: ACF window
  selection and benchmark metrics.

## 1. Initialize The Upstream Package

Initialize the vendored upstream package after cloning Detectiv:

```shell
git submodule update --init --recursive
```

This makes the upstream `TSB_AD` Python package available at
`external/tsb-ad`. Detectiv loads that exact checkout through `TSBADAdapter`.

## 2. Download And Place The Datasets

Download the datasets from the upstream project and extract them under `data/`:

1. Download [TSB-AD-U](https://www.thedatum.org/datasets/TSB-AD-U.zip) for
   univariate series.
2. Download [TSB-AD-M](https://www.thedatum.org/datasets/TSB-AD-M.zip) for
   multivariate series.
3. Extract each archive so its CSV files sit directly below the corresponding
   directory shown below.

Keep the upstream CSV filenames unchanged. Detectiv reads each filename's
embedded training boundary when it builds the temporal split.

Expected layout:

```text
data/
  TSB-AD-U/
    001_NAB_id_1_Facility_tr_1007_1st_2014.csv
    ...
  TSB-AD-M/
    001_SMD_id_1_machine-1-1_tr_2848_1st_2879.csv
    ...
```

The tracked directory placeholders establish this layout, but no benchmark CSV
data is committed to Detectiv.

## 3. Prepare One Series

This is the complete TSB-AD-specific setup for one experiment:

```python
from pathlib import Path

from detectiv.benchmarks.tsb_ad import (
    TSBADAdapter,
    TSBADCollection,
    TSBADCollectionLoader,
)
from detectiv.callbacks import MetricsCallback

loader = TSBADCollectionLoader(Path("data"))
split = loader.load_series(
    TSBADCollection.UNIVARIATE,
    "001_NAB_id_1_Facility_tr_1007_1st_2014",
)

adapter = TSBADAdapter(Path("external/tsb-ad"))
window_length = adapter.acf_window(split.train)
metrics = MetricsCallback(adapter.evaluator(sliding_window=window_length))
```

Use `split` as the input to the normal Detectiv preprocessing and image pipeline.
Pass `metrics` in the reconstruction scenario's `callbacks` argument. Labels
flow from the test series into the report, so they do not need to be supplied
again.

## 4. Stream A Collection

Iteration is deterministic by filename and loads one CSV only when requested:

```python
for split in loader.iter_collection(TSBADCollection.UNIVARIATE):
    run(split)
```

Filter a collection by its original source dataset when needed:

```python
groups = loader.source_groups(TSBADCollection.UNIVARIATE)
nab_splits = loader.iter_collection(
    TSBADCollection.UNIVARIATE,
    source_group="NAB",
)
```

## Evaluation Method

Detectiv deliberately fits preprocessing, projection, and the model on the train
prefix only. Compute ACF from `split.train`, then score and evaluate only the
held-out test suffix. This differs from upstream TSB-AD, which scores and
evaluates the full series including the training prefix. Detectiv results are
therefore not comparable to the upstream TSB-AD leaderboard.

Each report score array and its labels must have the same one-dimensional
shape. Labels must be binary and include both normal and anomalous points,
matching the requirements of the upstream metrics. The callback stores each
result as `<plan>.<propagation>.<metric>` in `report.metrics`.

## Integration Boundaries

- `load_tsb_ad_csv()` reads one benchmark CSV into a `TimeSeries`.
- `TSBADCollectionLoader` understands filenames, source-group metadata, and
  returns independent temporal splits without grouping files into datasets.
- `TSBADAdapter` owns the dependency on the upstream Python package and its
  metric functions.

Do not modify `external/tsb-ad`. It is a vendored submodule pinned by Detectiv;
keep local benchmark datasets under `data/` instead.
