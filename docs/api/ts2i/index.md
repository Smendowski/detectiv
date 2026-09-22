# TS2I Overview

The immutable fluent stages define the time-series-to-image flow:

1. `TimeSeries.split` returns `TimeSeriesSplit`.
2. Optionally accumulate preprocessing steps; each fits on the current training
   partition before transforming every partition.
3. `window` returns `WindowedTimeSeriesSplit` with per-partition specifications.
4. `project` returns `ProjectedImageStage`, which fits projection on train and
   generates referenced images for every partition.

`build` keeps images lazy in memory. `materialize` renders them once to an
atomically published NPY artifact and returns datasets backed by those files.

Projection has two independent choices. A channelization selects or derives
time-series planes; a transformation renders each plane as a two-dimensional
image. A scheme can use one transformation per plane or replicate one rendered
plane to the requested channel count.

::: detectiv.ts2i.preparation

::: detectiv.ts2i.image_source
