# API reference

Import public types from their domain namespace. Importing implementation modules is
not part of the supported API.

```python
from detectiv.benchmarks.tsb_ad import TSBADCsvLoader, TSBADRepository
from detectiv.callbacks import BaseCallback, EvaluationCallback
from detectiv.images import ImageDataset, ImageShape, TorchImageDataset
from detectiv.models.autoencoders import Autoencoder, AutoencoderTrainer
from detectiv.scenarios import ReconstructionScenario, RunArtifactWriter
from detectiv.scoring import ReconstructionScoringPlan
from detectiv.time_series import TemporalHoldout, TemporalSplitter, TimeSeries
from detectiv.time_series.windowing import TailPolicy, WindowSpec
from detectiv.ts2i import ImagePreparation, ImagePreparationInspection
```

Use the package's domain `__init__` modules as the stable import surface. The design
guide shows a complete TS2I workflow, and the benchmark reproduction scripts show
end-to-end scenario construction.
