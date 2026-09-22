# Callback Lifecycle

Callbacks observe a `ReconstructionScenario` without changing its training or
scoring configuration. The callback is supplied through the scenario's existing
`callbacks` argument:

```python
from detectiv.callbacks import RunArtifactCallback, TimingCallback

reporter = RunArtifactCallback("artifacts/my-run")
timer = TimingCallback()

scenario = ReconstructionScenario(
    images=images,
    model=model,
    training_mode=training_mode,
    scoring_plans=scoring_plans,
    callbacks=(timer, reporter),
)
result = scenario.run()

assert timer.elapsed_seconds is not None
assert reporter.artifacts is not None
```

The callback imports and constructor names above are public API. `images`,
`model`, `training_mode`, and `scoring_plans` are the inputs used to construct
the scenario.
Callback names must be unique, so register at most one instance of each built-in
callback type in a scenario.

## Lifecycle Semantics

The scenario applies its reproducibility settings, then calls
`on_run_started()` in registration order. It calls `on_epoch_finished(event)` in
that same order after every completed training epoch. After training, scoring,
and propagation succeed, it calls `on_run_finished(result)` in registration
order. A completion callback may return a same-type replacement result; the
scenario passes that enriched result to the next callback. Returning `None`
preserves the current result. `scenario.run()` returns the final result from this
ordered chain. It always calls `on_run_closed()` afterwards in reverse
registration order for callbacks whose start hook completed, allowing
resource-owning callbacks such as `MlflowCallback` to clean up reliably.

If startup, training, scoring, propagation, or an epoch hook raises, only
callbacks whose start hook already returned receive `on_run_failed(error)`, in
registration order. Exceptions raised while reporting failure are appended as
notes to the original error rather than replacing it. In contrast, an exception
from `on_run_finished()` propagates directly and does not cause failure hooks.

`TimingCallback` records the interval from its own start hook through its
successful or failed terminal hook. Its value is therefore unavailable before a
terminal hook, and registration position can slightly affect the measured
interval when other callbacks perform work.

## Choose A Callback

Use `TimingCallback` for a lightweight elapsed duration, including failed runs.
Read `elapsed_seconds` after `scenario.run()` returns or raises.

Use `MetricsCallback(evaluator)` to evaluate every reconstruction scoring plan,
propagation, and series after point scoring. The callback reads labels from the
`ReconstructionReport`, validates complete alignment before evaluation, and
returns metrics named `<plan>.<propagation>.<series>.<metric>`. Write the report
returned by `scenario.run()` to persist those metrics; do not pass labels to the
callback separately.

Use `RunArtifactCallback` to publish a local, verified run bundle only after
success. Supply `metrics_provider` for completion metrics, `provenance` for
metadata, and `visualize=True` when its optional visualization dependencies are
available. Its writer rejects an existing non-empty destination unless
`overwrite=True`.

Use `MlflowCallback` to track a scenario in MLflow. Install its optional
dependency first. The callback owns the MLflow run lifecycle. For reconstruction,
compose the generic and reconstruction-specific callbacks explicitly:

```console
uv sync --extra experiment
```

```python
from detectiv.callbacks import MlflowCallback, ReconstructionMlflowCallback
from detectiv.scenarios import ReconstructionScenario

tracking = MlflowCallback()
scenario = ReconstructionScenario(
    images=images,
    model=model,
    training_mode=training_mode,
    scoring_plans=scoring_plans,
    callbacks=(tracking, ReconstructionMlflowCallback(tracking)),
)
result = scenario.run()
```

The generic callback records lifecycle tags; the reconstruction extension records
epoch metrics, reports, curves, and optional model artifacts. Configure metadata
and metric providers on `MlflowCallback`; the [MLflow API](api/callbacks/mlflow.md)
defines generic tracking and the [reconstruction tracking API](api/callbacks/tracking/reconstruction.md)
defines reconstruction-specific behavior.

See the [base contract](api/callbacks/base.md),
[artifact callback](api/callbacks/artifacts.md), and
[timing callback](api/callbacks/timing.md) for exact signatures and contracts.
