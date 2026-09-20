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

The integration separates three concerns:

1. TSB-AD CSV files become typed Detectiv `TimeSeries` datasets with the
   benchmark's predefined temporal split.
2. `TSBADCollectionLoader` loads one named dataset directly or streams an
   entire collection without retaining every dataset in memory.
3. `TSBADAdapter` calls the upstream ACF window estimator and evaluation
   metrics without exposing upstream import details to an experiment.

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

## 3. Load One Dataset

Load one named dataset for an interactive experiment or a targeted benchmark:

```python
from pathlib import Path

from detectiv.benchmarks.tsb_ad import TSBADCollection, TSBADCollectionLoader

loader = TSBADCollectionLoader(Path("data"))
benchmark = loader.load_dataset(TSBADCollection.UNIVARIATE, "NAB")

dataset = benchmark.dataset
splitter = benchmark.splitter
```

`dataset` contains all series belonging to `NAB`. `splitter` carries the
per-series training boundaries parsed from the original TSB-AD filenames.

Use the same path for a multivariate dataset:

```python
benchmark = loader.load_dataset(TSBADCollection.MULTIVARIATE, "SMD")
```

## 4. Stream A Full Collection

For a complete collection, stream one logical dataset at a time. This avoids
retaining every TSB-AD dataset in memory:

```python
for benchmark in loader.iter_collection(TSBADCollection.UNIVARIATE):
    run(benchmark.dataset, benchmark.splitter)
```

Use `load_collection()` only when every dataset must be resident at once.

## 5. Evaluate Point Scores

`TSBADAdapter` is Detectiv's boundary to the initialized upstream package. It
provides the ACF window estimator and creates a configured evaluator for
validated point scores and binary labels:

```python
from pathlib import Path

from detectiv.benchmarks.tsb_ad import TSBADAdapter

adapter = TSBADAdapter(Path("external/tsb-ad"))
window = adapter.acf_window(dataset["001_NAB_id_1_Facility_tr_1007_1st_2014"])
evaluator = adapter.evaluator(sliding_window=window)
metrics = evaluator.evaluate(point_scores, labels)
```

`point_scores` and `labels` must have the same one-dimensional shape. Labels
must be binary and include both normal and anomalous points, matching the
requirements of the upstream metrics.

## Integration Boundaries

- `TSBADCsvLoader` reads one benchmark CSV into a `TimeSeries`.
- `TSBADCollectionLoader` understands the TSB-AD collection layout, groups
  series, and produces the temporal split.
- `TSBADAdapter` owns the dependency on the upstream Python package and its
  metric functions.

Do not modify `external/tsb-ad`. It is a vendored submodule pinned by Detectiv;
keep local benchmark datasets under `data/` instead.
