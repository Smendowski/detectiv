# Callback MLflow API

::: detectiv.callbacks.tracking.mlflow

`MlflowCallback` is result-independent. It publishes
common setup metadata, a Detectiv run ID, lifecycle tags, caller metrics, and
configured artifact directories. MLflow is loaded only when the scenario starts.

Framework-owned `detectiv.*` tags take precedence over user tags for the
reserved run identity, lifecycle, scenario type, lineage, and dataset manifest
fields. Lifecycle values are `started`, `succeeded`, and `failed`; failures also
record `detectiv.error_type`.

Use generic tracking with any scenario result type:

```python
scenario = CustomScenario(callbacks=(MlflowCallback(),))
result = scenario.run()
```

For a reconstruction scenario, use
`detectiv.callbacks.ReconstructionMlflowCallback` instead. It extends the
generic callback with reconstruction metrics and artifacts while owning the same
MLflow run lifecycle.
