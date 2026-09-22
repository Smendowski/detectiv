# Time Series To Images

TS2I turns chronological windows into channel-first image datasets for image
models. `ImagePreparation` keeps the process declarative: define the temporal
split, window specifications, optional preprocessing, and projection strategy,
then call `build` or `materialize`.

The fitted projection learns only from the training split. Each resulting image
keeps its `WindowReference`, so later reconstruction scores can be propagated
back to source time points.

See the [TS2I API reference](api/ts2i/index.md) for the complete pipeline and
available encodings.
