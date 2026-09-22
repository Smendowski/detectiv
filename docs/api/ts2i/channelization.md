# Channelization

A `Channelization` converts a window shaped `(time, features)` into one or more
planes. Projection schemes assign a transformation to each produced plane.

Use `IdentityChannelization` when a downstream transformation accepts the full
multivariate window. Feature, PCA, and statistical channelizations produce
univariate planes for transformations such as GAF, MTF, and recurrence plots.

`HighestVariabilityFeatureChannelization` normally selects features from the
training split, preventing test data from influencing representation selection.
Its `WINDOW` scope is intentionally per-window instead.

::: detectiv.ts2i.channelization.base

::: detectiv.ts2i.channelization.features

::: detectiv.ts2i.channelization.highest_variability

::: detectiv.ts2i.channelization.mean_std_max

::: detectiv.ts2i.channelization.pca
