# First Experiment

The first experiment is a complete Python example: temporal splitting, TS2I image
preparation, reconstruction training, score propagation, inspection, and a local
report bundle.

Run it from the repository root:

```console
uv run python examples/first_experiment.py
```

It prints preparation and scenario preflight summaries, then writes:

```text
artifacts/first_experiment/run.json
artifacts/first_experiment/point_scores.npz
```

The example uses synthetic labels only to define a visible anomalous region. They
are not used during training, calibration, or scoring.

Open `examples/first_experiment.py` and replace the synthetic `TimeSeriesDataset`,
projection, model, trainer, or scoring plan with your own Python objects. Attach
an `EvaluationCallback` only when point labels are available for final evaluation.

Pass `visualize=True` to `run()` after installing the optional experiment
dependencies to write training and score figures.

Inspect a completed local bundle from Python:

```python
from pathlib import Path

from detectiv.scenarios import RunArtifacts

artifacts = RunArtifacts.open(Path("artifacts/first_experiment"))
artifacts.verify()
report = artifacts.read_report()
scores = artifacts.load_scores()
```

`verify()` checks each artifact against the SHA-256 checksums recorded in
`run.json`. Loaded score arrays are immutable copies.
