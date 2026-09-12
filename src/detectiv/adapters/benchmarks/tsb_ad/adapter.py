import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np

from detectiv.evaluation import PointScoreEvaluator


def benchmark_acf_window(
    values: np.ndarray,
    *,
    repository: Path | None = None,
) -> int:
    find_length_rank, _ = _tsb_ad_functions(repository)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        feature = values
    elif values.ndim == 2 and values.shape[1] > 0:
        feature = values[:, 0]
    else:
        raise ValueError("values must be a non-empty one- or two-dimensional array")
    if len(feature) < 2 or not np.isfinite(feature).all():
        raise ValueError("values must contain at least two finite observations")
    return int(find_length_rank(feature.reshape(-1, 1), rank=1))


@dataclass(frozen=True)
class TSBADEvaluator(PointScoreEvaluator):
    sliding_window: int
    version: str = "opt"
    thresholds: int = 250
    repository: Path | None = None

    def __post_init__(self) -> None:
        if self.sliding_window <= 0 or self.thresholds <= 0:
            raise ValueError("sliding_window and thresholds must be positive")

    def evaluate(
        self, point_scores: np.ndarray, labels: np.ndarray
    ) -> dict[str, float]:
        _, get_metrics = _tsb_ad_functions(self.repository)
        scores, labels = _validate(point_scores, labels)
        metrics = get_metrics(
            scores,
            labels,
            slidingWindow=self.sliding_window,
            version=self.version,
            thre=self.thresholds,
        )
        return {name: float(value) for name, value in metrics.items()}


def _tsb_ad_functions(
    repository: Path | None,
) -> tuple[Callable[..., Any], Callable[..., dict[str, float]]]:
    root = repository or Path(__file__).resolve().parents[5]
    submodule = root / "external" / "tsb-ad"
    if not submodule.is_dir():
        raise FileNotFoundError("initialize the TSB-AD submodule before benchmarking")
    submodule_text = str(submodule)
    if submodule_text not in sys.path:
        sys.path.insert(0, submodule_text)
    metric_module = import_module("TSB_AD.evaluation.metrics")
    window_module = import_module("TSB_AD.utils.slidingWindows")
    return (
        cast(Callable[..., Any], window_module.find_length_rank),
        cast(Callable[..., dict[str, float]], metric_module.get_metrics),
    )


def _validate(
    point_scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(point_scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels):
        raise ValueError(
            "point_scores and labels must be one-dimensional with equal length"
        )
    if (
        not len(scores)
        or not np.isfinite(scores).all()
        or not np.isin(labels, (0, 1)).all()
    ):
        raise ValueError("scores must be finite and labels must be binary")
    if labels.min() == labels.max():
        raise ValueError("labels must contain both normal and anomalous points")
    return scores, labels
