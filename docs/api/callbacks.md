# Callbacks API

Scenarios invoke callbacks in registration order. A callback that completes
`on_run_started()` receives epoch, failure, and successful-completion hooks. If
an exception occurs during setup, training, scoring, or an epoch hook, only
already started callbacks receive `on_run_failed()`; notification errors do not
replace the primary error. A failure in `on_run_finished()` is propagated
directly and is not converted into a run-failure event.
