import ast
import importlib
from pathlib import Path

import pytest


def test_public_scenario_and_callback_namespaces_import_together() -> None:
    import detectiv.callbacks
    import detectiv.models
    import detectiv.protocols
    import detectiv.runs
    import detectiv.scenarios

    assert detectiv.callbacks.BaseCallback.__name__ == "BaseCallback"
    assert detectiv.models.Autoencoder.__name__ == "Autoencoder"
    assert (
        detectiv.protocols.SemiSupervisedTraining.__name__ == "SemiSupervisedTraining"
    )
    assert detectiv.runs.RunArtifactWriter.__name__ == "RunArtifactWriter"
    assert (
        detectiv.scenarios.ReconstructionScenario.__name__ == "ReconstructionScenario"
    )


@pytest.mark.parametrize(
    "module",
    (
        "detectiv.models.events",
        "detectiv.scenarios.artifacts",
        "detectiv.ts2i.projection.strategies.configured",
    ),
)
def test_obsolete_internal_modules_are_not_importable(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_domain_namespaces_do_not_reexport_other_domain_contracts() -> None:
    import detectiv.models
    import detectiv.scenarios

    assert not hasattr(detectiv.models, "TrainingEpochEvent")
    assert not hasattr(detectiv.scenarios, "ReproducibilitySettings")
    assert not hasattr(detectiv.scenarios, "RunArtifacts")
    assert not hasattr(detectiv.scenarios, "RunArtifactWriter")
    assert not hasattr(detectiv.scenarios, "SemiSupervisedTraining")
    assert not hasattr(detectiv.scenarios, "ValidationHoldout")


def test_scenarios_do_not_contain_policy_modules() -> None:
    source_root = Path(__file__).parents[2] / "src" / "detectiv" / "scenarios"

    assert not (source_root / "training" / "__init__.py").exists()
    assert not (source_root / "validation" / "__init__.py").exists()


def test_experiment_domain_dependencies_are_one_directional() -> None:
    source_root = Path(__file__).parents[2] / "src" / "detectiv"
    dependencies = {
        domain: _domain_dependencies(source_root / domain)
        for domain in ("callbacks", "models", "protocols", "runs", "scenarios")
    }

    assert dependencies["runs"] == set()
    assert dependencies["protocols"] == set()
    assert dependencies["models"] <= {"runs"}
    assert dependencies["callbacks"] <= {"runs"}
    assert dependencies["scenarios"] <= {"callbacks", "models", "protocols", "runs"}
    assert not _has_cycle(dependencies)


def test_generic_mlflow_tracker_has_no_concrete_scenario_dependencies() -> None:
    source = (
        Path(__file__).parents[2] / "src" / "detectiv" / "callbacks" / "mlflow.py"
    ).read_text(encoding="utf-8")

    forbidden = (
        "ReconstructionScenario",
        "ReconstructionScenarioResult",
        "TrainingEpochEvent",
        "torch",
        "numpy",
        "matplotlib",
        "RunArtifactWriter",
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
        if domain in {"callbacks", "models", "protocols", "runs", "scenarios"}
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
