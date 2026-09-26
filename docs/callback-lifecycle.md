# Callback Lifecycle

Callbacks observe a scenario without changing its training or scoring
configuration. They are supplied through the scenario's `callbacks` argument:

```python
from detectiv.callbacks import ReportArtifactCallback, TimingCallback

reporter = ReportArtifactCallback("artifacts/my-run")
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
resource-owning callbacks such as `MLflowCallback` to clean up reliably.

If startup, training, scoring, propagation, an epoch hook, or a completion hook
raises, only callbacks whose start hook already returned receive
`on_run_failed(error)`, in registration order. Exceptions raised while reporting
failure are appended as notes to the original error rather than replacing it.

`TimingCallback` records the interval from its own start hook through its
successful or failed terminal hook. Its value is therefore unavailable before a
terminal hook, and registration position can slightly affect the measured
interval when other callbacks perform work.

## Choose A Callback

Use `TimingCallback` for a lightweight elapsed duration, including failed runs.
Read `elapsed_seconds` after `scenario.run()` returns or raises.

Use `MetricsCallback(evaluator)` to evaluate every reconstruction scoring plan
and propagation after point scoring. The callback reads labels from the
`ReconstructionReport`, validates complete alignment before evaluation, and
returns metrics named `<plan>.<propagation>.<metric>`. Write the report
returned by `scenario.run()` to persist those metrics; do not pass labels to the
callback separately.

Use `ReportArtifactCallback` to publish a local, verified report bundle only
after success. It persists the completed report's metrics, so register
`MetricsCallback` before it when metrics should be included. Supply `provenance`
for metadata and `visualize=True` when its optional visualization dependencies
are available. Its writer rejects an existing non-empty destination unless
`overwrite=True`.

Use `MLflowTracker` to configure MLflow, then connect it to a scenario with an
MLflow callback. Install the optional dependency first:

```console
uv sync --extra experiment
```

```python
from detectiv.callbacks import MLflowCallback
from detectiv.scenarios import ReconstructionScenario
from detectiv.tracking import MLflowTracker

tracker = MLflowTracker(experiment_name="reconstruction")
tracking = MLflowCallback(tracker)
scenario = ReconstructionScenario(
    images=images,
    model=model,
    training_mode=training_mode,
    scoring_plans=scoring_plans,
    callbacks=(tracking,),
)
result = scenario.run()
```

`MLflowTracker` owns MLflow configuration and logging operations.
`BaseMLflowCallback` adapts generic scenario lifecycle events, while
`MLflowCallback` adds reconstruction epoch metrics, report metrics, and optional
model publication.

Compose callbacks in the order that data becomes available. Metrics enrich the
report first, the artifact callback writes that report, and MLflow uploads the
completed directory last:

```python
from pathlib import Path

from detectiv.callbacks import MLflowCallback, MetricsCallback, ReportArtifactCallback
from detectiv.tracking import MLflowTracker

report_directory = Path("artifacts/reconstruction")
callbacks = (
    MetricsCallback(evaluator),
    ReportArtifactCallback(report_directory, visualize=True),
    MLflowCallback(
        MLflowTracker(experiment_name="reconstruction"),
        artifact_directories={report_directory: "detectiv"},
    ),
)
```

The [tracking API](api/tracking/mlflow.md) documents MLflow setup, and the
[callback API](api/callbacks/mlflow.md) documents lifecycle adaptation.

See the [base contract](api/callbacks/base.md),
[artifact callback](api/callbacks/artifacts.md), and
[timing callback](api/callbacks/timing.md) for exact signatures and contracts.
