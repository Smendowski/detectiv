# Scenarios Overview

Scenarios orchestrate explicit scientific components into one reproducible run.
They do not define model behavior, training policy, scoring policy, or tracking:
those remain independently configured Python objects.

`BaseScenario` creates one Detectiv run identity, applies optional
reproducibility settings, dispatches callbacks, and exposes `completed_run`
after success. Callbacks receive setup, epoch, success or failure, and final
cleanup notifications in a defined order.

`ReconstructionScenario` is the current concrete scenario. It partitions image
windows using a training mode, trains an autoencoder, produces window evidence,
propagates it to time-point scores, and returns an immutable
`ReconstructionReport`. The report combines generic experiment metadata with
reconstruction-specific training and score data. Its
`inspect()` method reports inputs and configured plans without performing work.

- [Base scenario lifecycle](base.md)
- [Reconstruction scenario](reconstruction.md)
- [Experiment reports](results.md)
