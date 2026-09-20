# Experiment Surfaces

Detectiv's supported user surface is the installed Python API and the
copyable [first experiment](first-experiment.md). It writes a verified local run
bundle to an explicit directory and is covered by an integration test.

## Supported example

`examples/first_experiment.py` is an executable example for checkout users:

```console
uv run python examples/first_experiment.py
```

It requires only base dependencies and writes `run.json` and
`point_scores.npz` under `artifacts/first_experiment`. It uses seed 42 and
deterministic Torch algorithms; see the reproducibility limits in the first
experiment guide.

## Repository-only scripts

`tsb_ad_images.py` prepares image folders from a local TSB-AD CSV, and
`visualize_transformations.py` is exploratory tooling. They may change with
repository development and are not covered as release surfaces.

## Supported checkout reproduction

`scenarios/reproduce_prism_msl4.py` is a supported checkout reproduction for
one TSB-AD CSV and requires a local TSB-AD source checkout:

```console
uv run python scenarios/reproduce_prism_msl4.py path/to/series.csv \
  --tsb-ad-source external/tsb-ad \
  --output artifacts/prism-series \
  --seed 42
```

It writes a verified run bundle containing `run.json` and `point_scores.npz`,
then prints nested test metrics as JSON. The default seed is 42 and is recorded
with deterministic Torch settings in the bundle. It is a checkout executable,
not an installed package command or importable public API.

The TSB-AD integration resolves its repository path process-wide. One Python
process therefore cannot compare multiple TSB-AD repository revisions. Use a
separate process for each revision when comparing benchmark sources.
