# TS2I Overview

`ImagePreparation` is an immutable builder for the time-series-to-image flow:

1. Split a `TimeSeriesDataset` temporally.
2. Optionally fit preprocessing on the training partition.
3. Fit a `ProjectionStrategy` on training data.
4. Generate referenced image windows for each split.

`build` keeps images lazy in memory. `materialize` renders them once to an
atomically published NPY artifact and returns datasets backed by those files.

Projection has two independent choices. A channelization selects or derives
time-series planes; a transformation renders each plane as a two-dimensional
image. A scheme can use one transformation per plane or replicate one rendered
plane to the requested channel count.

::: detectiv.ts2i.preparation

::: detectiv.ts2i.image_source
