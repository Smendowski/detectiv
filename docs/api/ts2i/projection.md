# Projection

`ProjectionScheme` composes channelization with transformations. Call
`channels` with one transformation per channelization output, or call
`replicate` after configuring exactly one transformation to produce repeated
channels for a model that expects a fixed channel count.

`BaseProjectionStrategy.fit` receives training data and returns a fitted
`BaseProjectionScheme`. A scheme may render every window with one fixed
composition or select compatible projections dynamically per window.
`FixedProjectionStrategy` is suitable when every part of the scheme is already
configured or has no fit state.

::: detectiv.ts2i.projection.schemes.base

::: detectiv.ts2i.projection.strategies.base

::: detectiv.ts2i.projection.strategies.fixed
