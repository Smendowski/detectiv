# Scoring Overview

Scoring turns reconstruction behavior into anomaly evidence while preserving the
time-series location of every TS2I image window.

## Window Evidence

A `ReconstructionScorer` compares a reconstructed image with its input and
returns one value for each `WindowReference`. The built-in
`MeanSquaredWindowReconstructionError` uses elementwise mean squared error and
reduces it to one score per image window. Higher values indicate a less faithful
reconstruction and therefore stronger anomaly evidence.

`WindowScoreBatch` pairs those values with the original window references. This
association is essential: TS2I changes representation, but scoring still needs
to report evidence at the original time-series positions.

## Point Propagation

Windows commonly overlap, while anomaly detection ultimately needs one score per
source time point. A `PointScoringPlan` has two explicit steps:

1. A `PointAssignment` distributes each window's evidence across its valid
   points. `UniformPointAssignment` gives every valid point the window score.
2. A `PointScoreAggregator` combines contributions from overlapping windows.
   Mean, maximum, and median aggregation are provided.

The result is one score per time point for each series. Scores are never mixed
between series.

## Coverage

Every point should normally be covered by at least one window. The default
`UncoveredPolicy.ERROR` rejects an incomplete window configuration. Use
`UncoveredPolicy.EDGE_PAD` only when trailing points are intentionally uncovered:
it repeats the final covered score across that suffix.

## Plans

`ReconstructionScoringPlan` combines one reconstruction scorer with one or more
point-scoring plans. This makes alternative propagation policies explicit and
comparable without rerunning reconstruction inference.

See the API reference for exact contracts:

- [Reconstruction scorers](reconstruction.md)
- [Window evidence](window-scores.md)
- [Propagation](propagation.md)
- [Scoring plans](plans.md)
