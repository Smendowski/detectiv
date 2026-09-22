# Experiment Reports

Every scenario returns a scenario-specific `ExperimentReport`. Common report
behavior includes input provenance, reproducibility, performance metadata,
evaluation metrics, human-readable summaries, and serializable records.

`ReconstructionReport` additionally exposes training history, window evidence,
and point scores. Use `point_scores_for()` for visualization and analysis, and
`write()` to persist report metadata and arrays as a verified run artifact. Its
return value provides the `run.json` and point-score paths for downstream tools;
`summary()` presents qualified metric names compactly without changing their
machine-readable keys.

::: detectiv.scenarios.results
