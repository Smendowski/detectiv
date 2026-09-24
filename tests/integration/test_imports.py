import ast
import importlib
from pathlib import Path

import pytest


def test_public_scenario_and_callback_namespaces_import_together() -> None:
    import detectiv.callbacks
    import detectiv.models
    import detectiv.protocols
    import detectiv.reports
    import detectiv.reproducibility
    import detectiv.runs
    import detectiv.scenarios

    assert detectiv.callbacks.BaseCallback.__name__ == "BaseCallback"
    assert detectiv.models.Autoencoder.__name__ == "Autoencoder"
    assert (
        detectiv.protocols.SemiSupervisedTraining.__name__ == "SemiSupervisedTraining"
    )
    assert (
        detectiv.reports.ReconstructionReportWriter.__name__
        == "ReconstructionReportWriter"
    )
    assert (
        detectiv.reproducibility.ReproducibilitySettings.__name__
        == "ReproducibilitySettings"
    )
    assert (
        detectiv.reproducibility.configure_reproducibility.__name__
        == "configure_reproducibility"
    )
    assert (
        detectiv.scenarios.ReconstructionScenario.__name__ == "ReconstructionScenario"
    )


@pytest.mark.parametrize(
    "module",
    (
        "detectiv.models.events",
        "detectiv.runs.artifacts",
        "detectiv.runs.reproducibility",
        "detectiv.scenarios.artifacts",
        "detectiv.scenarios.results",
        "detectiv.ts2i.projection.strategies.configured",
    ),
)
def test_obsolete_internal_modules_are_not_importable(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_domain_namespaces_do_not_reexport_other_domain_contracts() -> None:
    import detectiv.models
    import detectiv.runs
    import detectiv.scenarios

    assert not hasattr(detectiv.models, "TrainingEpochEvent")
    assert not hasattr(detectiv.scenarios, "ReproducibilitySettings")
    assert not hasattr(detectiv.runs, "ReportArtifacts")
    assert not hasattr(detectiv.runs, "ReconstructionReportWriter")
    assert not hasattr(detectiv.runs, "ReproducibilitySettings")
    assert not hasattr(detectiv.runs, "configure_reproducibility")
    assert not hasattr(detectiv.scenarios, "ExperimentReport")
    assert not hasattr(detectiv.scenarios, "ReconstructionReport")
    assert not hasattr(detectiv.scenarios, "ReconstructionScenarioResult")
    assert not hasattr(detectiv.scenarios, "SemiSupervisedTraining")
    assert not hasattr(detectiv.scenarios, "ValidationHoldout")


def test_time_series_namespace_excludes_removed_multi_series_contracts() -> None:
    import detectiv.time_series
    import detectiv.time_series.windowing
    import detectiv.ts2i

    assert not hasattr(detectiv.time_series, "TimeSeriesDataset")
    assert not hasattr(detectiv.time_series, "TemporalSplitter")
    assert detectiv.time_series.TimeSeriesSplit.__name__ == "TimeSeriesSplit"
    assert (
        detectiv.time_series.windowing.WindowedTimeSeriesSplit.__name__
        == "WindowedTimeSeriesSplit"
    )
    assert not hasattr(detectiv.ts2i, "ImagePreparation")
    assert detectiv.ts2i.ProjectedImageStage.__name__ == "ProjectedImageStage"


def test_time_series_package_has_no_ts2i_imports() -> None:
    source_root = Path(__file__).parents[2] / "src" / "detectiv" / "time_series"

    imported_modules = {
        module for path in source_root.rglob("*.py") for module in _all_imports(path)
    }

    assert not any(
        module == "detectiv.ts2i" or module.startswith("detectiv.ts2i.")
        for module in imported_modules
    )


def test_runs_package_has_no_reproducibility_imports() -> None:
    source_root = Path(__file__).parents[2] / "src" / "detectiv" / "runs"

    imported_modules = {
        module for path in source_root.rglob("*.py") for module in _all_imports(path)
    }

    assert not any(
        module == "detectiv.reproducibility"
        or module.startswith("detectiv.reproducibility.")
        for module in imported_modules
    )


def test_scenarios_do_not_contain_policy_modules() -> None:
    source_root = Path(__file__).parents[2] / "src" / "detectiv" / "scenarios"

    assert not (source_root / "training" / "__init__.py").exists()
    assert not (source_root / "validation" / "__init__.py").exists()


def test_experiment_domain_dependencies_are_one_directional() -> None:
    source_root = Path(__file__).parents[2] / "src" / "detectiv"
    dependencies = {
        domain: _domain_dependencies(source_root / domain)
        for domain in (
            "callbacks",
            "models",
            "protocols",
            "reports",
            "runs",
            "scenarios",
        )
    }

    assert dependencies["runs"] == set()
    assert dependencies["protocols"] == set()
    assert dependencies["models"] <= {"runs"}
    assert dependencies["reports"] <= {"runs"}
    assert dependencies["callbacks"] <= {"reports", "runs"}
    assert dependencies["scenarios"] <= {
        "callbacks",
        "models",
        "protocols",
        "reports",
        "runs",
    }
    assert not _has_cycle(dependencies)


def test_generic_mlflow_tracker_has_no_concrete_scenario_dependencies() -> None:
    source = (
        Path(__file__).parents[2]
        / "src"
        / "detectiv"
        / "callbacks"
        / "tracking"
        / "mlflow.py"
    ).read_text(encoding="utf-8")

    forbidden = (
        "ReconstructionScenario",
        "ReconstructionReport",
        "TrainingEpochEvent",
        "torch",
        "numpy",
        "matplotlib",
        "ReconstructionReportWriter",
        "model_type",
    )
    assert not any(name in source for name in forbidden)


def _domain_dependencies(directory: Path) -> set[str]:
    dependencies: set[str] = set()
    for path in directory.rglob("*.py"):
        dependencies.update(_imports_from(path))
    dependencies.discard(directory.name)
    return dependencies


def _imports_from(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    visitor = _ImportVisitor()
    visitor.visit(tree)
    return visitor.imports


def _all_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


class _ImportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: set[str] = set()

    def visit_If(self, node: ast.If) -> None:
        if not _is_type_checking_guard(node):
            self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is not None:
            self.imports.update(_experiment_domains(node.module))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.update(_experiment_domains(alias.name))


def _is_type_checking_guard(node: ast.If) -> bool:
    return isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"


def _experiment_domains(module: str) -> set[str]:
    prefix = "detectiv."
    if not module.startswith(prefix):
        return set()
    domain = module.removeprefix(prefix).split(".", maxsplit=1)[0]
    return (
        {domain}
        if domain
        in {"callbacks", "models", "protocols", "reports", "runs", "scenarios"}
        else set()
    )


def _has_cycle(dependencies: dict[str, set[str]]) -> bool:
    visited: set[str] = set()
    active: set[str] = set()

    def visit(domain: str) -> bool:
        if domain in active:
            return True
        if domain in visited:
            return False
        visited.add(domain)
        active.add(domain)
        cyclic = any(visit(dependency) for dependency in dependencies[domain])
        active.remove(domain)
        return cyclic

    return any(visit(domain) for domain in dependencies)
