# Transformations

A `TS2ITransformation` renders a univariate or multivariate plane into one
image channel. Its `input_kinds` declare accepted plane shapes, which
`ProjectionScheme` validates before rendering. All transformations return the
requested `(height, width)` image size.

The registry supports constructing built-in and application-defined
transformations by their registered names.

::: detectiv.ts2i.transformations.base

::: detectiv.ts2i.transformations.registry

## Matrix Encodings

`GASF` and `GADF` are Gramian angular fields. `MTF` is a Markov transition
field, and `RP` is a recurrence plot. These accept univariate values.

::: detectiv.ts2i.transformations.gaf

::: detectiv.ts2i.transformations.mtf

::: detectiv.ts2i.transformations.rp

## Spatial Encodings

`LinePlot` renders a univariate trace. `LineGrid` tiles multivariate traces,
`StateGrid` resizes values directly, and `Spiral` maps a univariate trace onto a
spiral-like radial layout.

::: detectiv.ts2i.transformations.line_plot

::: detectiv.ts2i.transformations.line_grid

::: detectiv.ts2i.transformations.state_grid

::: detectiv.ts2i.transformations.spiral

## Wavelets And Baseline

`RWT` and `MWT` produce continuous wavelet scalograms with Ricker and Morlet
wavelets. `RandomNoise` is a seeded stochastic baseline useful for pipeline
checks, not a representation of input values.

::: detectiv.ts2i.transformations.wavelets

::: detectiv.ts2i.transformations.random_noise
