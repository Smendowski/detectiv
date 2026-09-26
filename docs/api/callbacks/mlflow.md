# MLflow Callbacks

## Lifecycle Adapter

::: detectiv.callbacks.BaseMLflowCallback

`BaseMLflowCallback` connects any scenario result type to an
`MLflowTracker`. It records Detectiv run identity and status, and can publish
caller-provided final metrics and existing artifact directories.

## Reconstruction Callback

::: detectiv.callbacks.MLflowCallback

`MLflowCallback` extends the lifecycle adapter with reconstruction epoch
metrics, training summary tags, report metrics, and optional model publication.
Report bundles and plots remain the responsibility of `ReportArtifactCallback`.
