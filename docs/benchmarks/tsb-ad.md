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

1. Each TSB-AD CSV becomes one independent `TSBADDataset` run unit containing
   one `TimeSeries` and one temporal boundary.
2. `TSBADCollectionLoader` loads one exact series or lazily streams CSV files.
   Names such as NAB and SMD are source metadata groups used for filtering.
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

## 3. Load One Series

Load one exact CSV series for an interactive experiment:

```python
from pathlib import Path

from detectiv.benchmarks.tsb_ad import TSBADCollection, TSBADCollectionLoader

loader = TSBADCollectionLoader(Path("data"))
series_id = "001_NAB_id_1_Facility_tr_1007_1st_2014"
benchmark = loader.load_series(TSBADCollection.UNIVARIATE, series_id)

series = benchmark.series
boundary = benchmark.boundary
```

Discover or filter source groups without combining their files:

```python
groups = loader.source_groups(TSBADCollection.UNIVARIATE)
for benchmark in loader.iter_collection(TSBADCollection.UNIVARIATE, source_group="NAB"):
    inspect(benchmark.series, benchmark.boundary)
```

## 4. Stream A Collection

Iteration is deterministic by filename and loads one CSV only when requested:

```python
for benchmark in loader.iter_collection(TSBADCollection.UNIVARIATE):
    run(benchmark.series, benchmark.boundary)
```

## 5. Evaluate Point Scores

`TSBADAdapter` is Detectiv's boundary to the initialized upstream package. It
provides the ACF window estimator and creates a configured evaluator. Register
that evaluator as a scenario callback; labels flow from the source `TimeSeries`
into the reconstruction report and are not passed again:

```python
from pathlib import Path

from detectiv.benchmarks.tsb_ad import TSBADAdapter
from detectiv.callbacks import MetricsCallback

adapter = TSBADAdapter(Path("external/tsb-ad"))
split = series.split(boundary)
window = adapter.acf_window(split.train)
evaluator = adapter.evaluator(sliding_window=window)
scenario = ReconstructionScenario(
    # ...
    callbacks=(MetricsCallback(evaluator),),
)
report = scenario.run()
artifacts = report.write(output_directory)
print(report.summary())
print(artifacts.manifest)
print(artifacts.point_scores)
```

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

- `TSBADCsvLoader` reads one benchmark CSV into a `TimeSeries`.
- `TSBADCollectionLoader` understands filenames, source-group metadata, and
  temporal boundaries without grouping files into training datasets.
- `TSBADAdapter` owns the dependency on the upstream Python package and its
  metric functions.

Do not modify `external/tsb-ad`. It is a vendored submodule pinned by Detectiv;
keep local benchmark datasets under `data/` instead.
