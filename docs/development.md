# Development

Create the project environment and install the Git hooks:

```console
just bootstrap
```

Run the complete local validation suite:

```console
just check
```

## Experiment tracking

Start the local MLflow tracking server and open <http://127.0.0.1:5000>:

```console
docker compose up -d mlflow
```

The service is deliberately bound to loopback only. Its SQLite metadata store and
artifacts persist in the `mlflow-data` Docker volume. Stop it without deleting
experiments with:

```console
docker compose down
```

Install the optional MLflow client for native runs:

```console
uv sync --extra experiment
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Keep MLflow at the scenario boundary. `MlflowExperiment` provides an explicit
run lifecycle and callback for per-epoch loss, system metrics, configuration,
parameters, final metrics, and run artifacts:

```python
from pathlib import Path

from detectiv.callbacks import MlflowExperiment, MlflowModelLogging
from detectiv.scenarios import RunArtifactWriter

tracker = MlflowExperiment(
    "tsb-ad/nab",
    run_name="001_NAB_id_1_Facility / RandomNoise / CNN / seed=42",
    parameters={"trainer": {"epochs": 100, "learning_rate": 1e-3}},
    configuration={"protocol": "temporal-validation-v1"},
    dataset={
        "name": "NAB",
        "series": ["001_NAB_id_1_Facility"],
        "source_sha256": "raw-data-content-hash",
    },
    tags={
        "dataset.name": "NAB",
        "model.family": "CNN",
        "series": "001_NAB_id_1_Facility",
    },
    description="Temporal-validation reconstruction baseline.",
    model=MlflowModelLogging(name="reconstruction_model"),
)
with tracker:
    result = scenario.run()
    artifacts = RunArtifactWriter(Path("artifacts/MSL5")).write(result)
    tracker.log_artifacts(artifacts.manifest.parent)
    tracker.log_metrics(final_metrics)
```

Pass `nested=True` when opening per-series runs beneath a study run.
Include a raw source checksum in every dataset manifest. MLflow records a
canonical manifest hash as `detectiv.dataset.manifest_sha256` for filtering and
reproducibility.
Set `registered_model_name` only for an intentionally promoted final model; it
creates a new Model Registry version rather than registering every benchmark run.

### UI organization

Use one experiment per dataset, not per series, seed, or date. For example:
`tsb-ad/nab`. This keeps the comparison table useful across all trials for that
dataset.

Use nested grouping runs for the dataset's series and TS2I transform. The model
leaf is one concrete training execution:

```text
tsb-ad/nab
  001_NAB_id_1_Facility
    RandomNoise
      CNN  [seed=42, run_id=MLflow-generated UUID]
```

Use simple names for tree nodes. Store details such as seed, architecture
parameters, and resolved configuration in tags, parameters, and artifacts. A
repeated `CNN` run is a separate execution with its own MLflow `run_id`; do not
add an application-specific execution UUID.

Use these tags consistently for UI filters:

```text
run.kind: group | training | tuning | ablation | smoke
dataset.name: NAB
model.family: CNN
series: 001_NAB_id_1_Facility
transform.name: RandomNoise
seed: 42
```

Use metric namespaces consistently:

```text
training.*
validation.*
test.*
system/*
```

Series and TS2I runs are grouping-only containers. They have no model, telemetry,
artifacts, or metrics. The model execution leaf exposes `training.*`,
`validation.*`, `test.*`, `system/*`, the final model, and
`detectiv/training_loss.png`, which overlays training and validation loss.
Final score keys include their scoring plan and aggregation, such as
`test.mean_squared_window.uniform_mean.VUS-PR`.
Training epoch count and best-loss summaries are stored as run tags rather than
one-point metrics.

Keep immutable run inputs under `detectiv/` artifacts:

```text
detectiv/configuration.json
detectiv/dataset.json
detectiv/run.json
detectiv/point_scores.npz
```

### Local reports

`RunArtifactWriter` creates a versioned `run.json` report and compressed point
scores. The report records runtime provenance, optional caller-supplied
provenance, training summary, evaluation metrics, and SHA-256 checksums for all
generated artifacts.

Attach `RunArtifactCallback` to write the same bundle automatically when a
scenario completes:

```python
from pathlib import Path

from detectiv.callbacks import RunArtifactCallback

reporter = RunArtifactCallback(
    Path("runs/nab-cnn"),
    provenance={"seed": 42},
    visualize=True,
)
```

Set `visualize=True` after installing the `experiment` extra to generate a
combined training-loss figure and score timelines under `figures/`.

## Devcontainer

The devcontainer starts the same Compose project, installs locked development
and optional dependencies, and uses `http://mlflow:5000` inside the Compose
network. Open the repository with a Dev Containers-compatible editor and select
"Reopen in Container".

It standardizes Linux development tools and CI-like checks. Run macOS MPS
training natively instead; Linux CUDA training can be added as a separate
accelerator-specific profile.
