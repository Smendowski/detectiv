import runpy
import shutil
from pathlib import Path

import pytest


def test_synthetic_reconstruction_import_has_no_execution_side_effects(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = Path(__file__).parents[2] / "samples" / "01_synthetic_reconstruction.py"
    removed: list[Path] = []

    def record_removal(path: Path, *, ignore_errors: bool = False) -> None:
        removed.append(path)

    monkeypatch.setattr(shutil, "rmtree", record_removal)

    namespace = runpy.run_path(str(sample), run_name="imported_sample")

    assert namespace["OUTPUT_DIRECTORY"] == (
        sample.parents[1] / "outputs" / sample.stem
    )
    assert callable(namespace["main"])
    assert "TimingCallback" in namespace
    assert "MetricsCallback" not in namespace
    assert "TSBADAdapter" not in namespace
    assert removed == []
    assert capsys.readouterr().out == ""
