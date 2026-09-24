# API Reference

Detectiv's API reference follows the public package layout under
`src/detectiv/`. Import types from their domain namespace rather than an
implementation module.

```python
from detectiv.benchmarks.tsb_ad import TSBADAdapter, TSBADCsvLoader
from detectiv.callbacks import (
    BaseCallback,
    MlflowCallback,
    ReportArtifactCallback,
)
from detectiv.images import ImageDataset, ImageShape, TorchImageDataset
from detectiv.models.autoencoders import Autoencoder, AutoencoderTrainer
from detectiv.protocols import RandomHoldout, SemiSupervisedTraining
from detectiv.reports import (
    ExperimentReport,
    ReconstructionReport,
    ReconstructionReportWriter,
    ReportArtifacts,
)
from detectiv.reproducibility import ReproducibilitySettings, configure_reproducibility
from detectiv.scenarios import ReconstructionScenario
from detectiv.scoring import ReconstructionScoringPlan
from detectiv.time_series import (
    TemporalBoundary,
    TemporalHoldout,
    TimeSeries,
    TimeSeriesSplit,
)
from detectiv.time_series.windowing import (
    TailPolicy,
    WindowedTimeSeriesSplit,
    WindowProjection,
    WindowSpec,
)
from detectiv.ts2i import ProjectedImageInspection, ProjectedImageStage
from detectiv.ts2i.projection import FixedProjectionStrategy, ProjectionScheme
```

API pages are added beside the corresponding public package as its contracts
are documented. This keeps each reference page small and aligned with the
repository structure.
