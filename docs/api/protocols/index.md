# Protocols Overview

Protocols make experimental choices explicit rather than hiding them in a
scenario or trainer.

`TrainingMode` selects the image windows used to fit a model. The built-in
`SemiSupervisedTraining` selects only windows labeled normal. A training mode
then returns a `TrainingPartition` containing training indices and optional
validation images or indices.

Validation may come from a separate temporal image split or a `ValidationHoldout`
applied to selected training windows. `RandomHoldout` is reproducible but rejects
overlapping windows because random splitting would leak the same source points
between training and validation.

- [Training](training.md)
- [Validation](validation.md)
