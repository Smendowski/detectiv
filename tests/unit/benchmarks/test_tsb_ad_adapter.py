import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from detectiv.benchmarks.tsb_ad import TSBADAdapter
from detectiv.time_series import TimeSeries


@pytest.fixture(autouse=True)
def _unload_tsb_ad() -> Iterator[None]:
    _remove_tsb_ad_modules()
    yield
    _remove_tsb_ad_modules()


def test_adapter_reuses_upstream_functions_without_changing_sys_path(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    adapter = TSBADAdapter(source)
    path_before = tuple(sys.path)

    window = adapter.acf_window(
        TimeSeries(np.array([[1.0, 2.0], [3.0, 4.0]])), feature_index=1
    )
    metrics = adapter.evaluator(sliding_window=window, thresholds=7).evaluate(
        np.array([0.1, 0.9]), np.array([0, 1])
    )

    assert window == 6
    assert metrics == {"thresholds": 7.0, "window": 6.0}
    assert tuple(sys.path) == path_before


def test_adapter_rejects_invalid_source_directories(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="must contain"):
        TSBADAdapter(tmp_path)


def test_adapter_validates_the_acf_feature_index(tmp_path: Path) -> None:
    adapter = TSBADAdapter(_source(tmp_path))
    series = TimeSeries(np.array([[1.0], [2.0]]))

    with pytest.raises(ValueError, match="feature_index"):
        adapter.acf_window(series, feature_index=1)


@pytest.mark.parametrize("feature_index", [0.5, True])
def test_adapter_requires_an_integral_acf_feature_index(
    tmp_path: Path, feature_index: object
) -> None:
    adapter = TSBADAdapter(_source(tmp_path))

    with pytest.raises(ValueError, match="feature_index must be an integer"):
        adapter.acf_window(
            TimeSeries(np.array([[1.0], [2.0]])),
            feature_index=feature_index,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("sliding_window", [1.5, True])
def test_evaluator_requires_integral_positive_configuration(
    tmp_path: Path, sliding_window: object
) -> None:
    with pytest.raises(ValueError, match="sliding_window must be a positive integer"):
        TSBADAdapter(_source(tmp_path)).evaluator(sliding_window=sliding_window)  # type: ignore[arg-type]


def test_evaluator_rejects_labels_before_calling_tsb_ad(tmp_path: Path) -> None:
    evaluator = TSBADAdapter(_source(tmp_path)).evaluator(sliding_window=1)

    with pytest.raises(ValueError, match="boolean or integer"):
        evaluator.evaluate(np.array([0.1, 0.9]), np.array([0.2, 1.8]))


@pytest.mark.parametrize("labels", (np.array([False, True]), np.array([0, 1])))
def test_evaluator_accepts_boolean_and_integer_labels(
    tmp_path: Path, labels: np.ndarray
) -> None:
    evaluator = TSBADAdapter(_source(tmp_path)).evaluator(sliding_window=1)

    metrics = evaluator.evaluate(np.array([0.1, 0.9]), labels)

    assert metrics == {"thresholds": 250.0, "window": 1.0}


@pytest.mark.parametrize("labels", (np.array([0.0, 1.0]), np.array(["0", "1"])))
def test_evaluator_rejects_non_integer_label_representations(
    tmp_path: Path, labels: np.ndarray
) -> None:
    evaluator = TSBADAdapter(_source(tmp_path)).evaluator(sliding_window=1)

    with pytest.raises(ValueError, match="boolean or integer"):
        evaluator.evaluate(np.array([0.1, 0.9]), labels)


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "tsb-ad"
    package = source / "TSB_AD"
    evaluation = package / "evaluation"
    utilities = package / "utils"
    evaluation.mkdir(parents=True)
    utilities.mkdir()
    _write(package / "__init__.py", "")
    _write(evaluation / "__init__.py", "")
    _write(utilities / "__init__.py", "")
    _write(
        evaluation / "metrics.py",
        "def get_metrics(score, labels, slidingWindow, version, thre):\n"
        "    return {'window': slidingWindow, 'thresholds': thre}\n",
    )
    _write(
        utilities / "slidingWindows.py",
        "def find_length_rank(data, rank=1):\n    return int(data.sum())\n",
    )
    return source


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _remove_tsb_ad_modules() -> None:
    for name in tuple(sys.modules):
        if name == "TSB_AD" or name.startswith("TSB_AD."):
            del sys.modules[name]
