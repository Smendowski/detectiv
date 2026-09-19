import runpy
from pathlib import Path


def test_first_experiment_runs_and_writes_a_report(tmp_path: Path) -> None:
    module = runpy.run_path("examples/first_experiment.py")

    result = module["run"](tmp_path)

    assert len(result.training_losses) == 3
    assert (tmp_path / "run.json").is_file()
    assert (tmp_path / "point_scores.npz").is_file()
