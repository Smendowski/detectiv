# API Reference

Detectiv's API reference follows the public package layout under
`src/detectiv/`. Import types from their domain namespace rather than an
implementation module.

```python
from detectiv.benchmarks.tsb_ad import TSBADAdapter, TSBADCsvLoader
from detectiv.callbacks import BaseCallback, MlflowCallback, RunArtifactCallback
from detectiv.images import ImageDataset, ImageShape, TorchImageDataset
from detectiv.models.autoencoders import Autoencoder, AutoencoderTrainer
from detectiv.protocols import RandomHoldout, SemiSupervisedTraining
from detectiv.runs import ReproducibilitySettings, RunArtifacts, RunArtifactWriter
from detectiv.scenarios import (
    ReconstructionScenario,
    ReconstructionScenarioResult,
)
from detectiv.scoring import ReconstructionScoringPlan
from detectiv.time_series import TemporalHoldout, TemporalSplitter, TimeSeries
from detectiv.time_series.windowing import TailPolicy, WindowSpec
from detectiv.ts2i import ImagePreparation, ImagePreparationInspection
from detectiv.ts2i.projection import FixedProjectionStrategy, ProjectionScheme
```

API pages are added beside the corresponding public package as its contracts
are documented. This keeps each reference page small and aligned with the
repository structure.
