# Time Series To Images

TS2I turns chronological windows into channel-first image datasets for image
models. The fluent stages keep the process declarative: split a `TimeSeries`,
optionally configure preprocessing, define partition window specifications, and
select a projection strategy before calling `build`, `inspect`, or `materialize`.
Repeated `preprocess` calls accumulate in declaration order; each step fits only
on the currently transformed training partition.

The fitted projection learns only from the training split. Each resulting image
keeps its `WindowReference`, so later reconstruction scores can be propagated
back to source time points.

See the [TS2I API reference](api/ts2i/index.md) for the complete pipeline and
available encodings.
