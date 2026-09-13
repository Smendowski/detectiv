import sys
from dataclasses import dataclass
from importlib import import_module, util
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

from detectiv.time_series import TimeSeries

if TYPE_CHECKING:
    from detectiv.benchmarks.tsb_ad.evaluation import TSBADEvaluator


@dataclass(frozen=True)
class TSBADRepository:
    source_directory: Path

    def __post_init__(self) -> None:
        source_directory = self.source_directory.resolve()
        package_directory = source_directory / "TSB_AD"
        if (
            not package_directory.is_dir()
            or not (package_directory / "__init__.py").is_file()
        ):
            raise FileNotFoundError(
                "TSB-AD source directory must contain the TSB_AD package: "
                f"{source_directory}"
            )
        object.__setattr__(self, "source_directory", source_directory)

    def acf_window(self, series: TimeSeries, *, feature_index: int = 0) -> int:
        if not 0 <= feature_index < series.n_features:
            raise ValueError("feature_index must identify a feature in the series")
        if series.n_timesteps < 2:
            raise ValueError("series must contain at least two observations")
        feature = series.values[:, feature_index]
        find_length_rank, _ = self._functions()
        return int(find_length_rank(feature.reshape(-1, 1), rank=1))

    def evaluator(
        self,
        *,
        sliding_window: int,
        version: str = "opt",
        thresholds: int = 250,
    ) -> "TSBADEvaluator":
        from detectiv.benchmarks.tsb_ad.evaluation import TSBADEvaluator

        return TSBADEvaluator(
            repository=self,
            sliding_window=sliding_window,
            version=version,
            thresholds=thresholds,
        )

    def _metrics(
        self,
        point_scores: np.ndarray,
        labels: np.ndarray,
        *,
        sliding_window: int,
        version: str,
        thresholds: int,
    ) -> dict[str, float]:
        _, get_metrics = self._functions()
        metrics = get_metrics(
            point_scores,
            labels,
            slidingWindow=sliding_window,
            version=version,
            thre=thresholds,
        )
        return {name: float(value) for name, value in metrics.items()}

    def _functions(self) -> tuple[Any, Any]:
        self._package()
        metric_module = import_module("TSB_AD.evaluation.metrics")
        window_module = import_module("TSB_AD.utils.slidingWindows")
        return window_module.find_length_rank, metric_module.get_metrics

    def _package(self) -> ModuleType:
        package_directory = self.source_directory / "TSB_AD"
        existing = sys.modules.get("TSB_AD")
        if existing is not None:
            if _package_directory(existing) != package_directory:
                raise RuntimeError(
                    "TSB_AD is already loaded from a different source directory"
                )
            return existing

        specification = util.spec_from_file_location(
            "TSB_AD",
            package_directory / "__init__.py",
            submodule_search_locations=[str(package_directory)],
        )
        if specification is None or specification.loader is None:
            raise ImportError(f"unable to load TSB_AD from {self.source_directory}")
        package = util.module_from_spec(specification)
        sys.modules["TSB_AD"] = package
        try:
            specification.loader.exec_module(package)
        except BaseException:
            del sys.modules["TSB_AD"]
            raise
        return package


def _package_directory(package: ModuleType) -> Path | None:
    locations = getattr(package, "__path__", ())
    for location in locations:
        return Path(location).resolve()
    return None
