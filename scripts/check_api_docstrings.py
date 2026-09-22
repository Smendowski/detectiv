#!/usr/bin/env python3
"""Audit public Python API docstrings and their Google-style sections."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Section = Literal["Args", "Returns", "Raises"]


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    symbol: str
    message: str


def public_name(name: str) -> bool:
    """Return whether a definition is part of the public API."""
    return not name.startswith("_")


def has_section(docstring: str, section: Section) -> bool:
    """Return whether a Google-style section heading appears in a docstring."""
    return any(line.strip() == f"{section}:" for line in docstring.splitlines())


def needs_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a callable accepts user-visible arguments."""
    arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
    return any(argument.arg not in {"self", "cls"} for argument in arguments) or bool(
        node.args.vararg or node.args.kwarg
    )


def needs_returns(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a callable declares a non-None return annotation."""
    if node.returns is None:
        return False
    return not (isinstance(node.returns, ast.Constant) and node.returns.value is None)


def needs_raises(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a callable directly raises an exception."""
    return any(isinstance(child, ast.Raise) for child in ast.walk(node))


def audit_callable(
    path: Path,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    symbol: str,
) -> list[Finding]:
    """Return missing-docstring and missing-section findings for one callable."""
    docstring = ast.get_docstring(node)
    if docstring is None:
        return [Finding(path, node.lineno, symbol, "missing docstring")]

    findings: list[Finding] = []
    required: tuple[tuple[Section, bool], ...] = (
        ("Args", needs_args(node)),
        ("Returns", needs_returns(node)),
        ("Raises", needs_raises(node)),
    )
    for section, required_here in required:
        if required_here and not has_section(docstring, section):
            findings.append(
                Finding(path, node.lineno, symbol, f"missing {section} section")
            )
    return findings


def audit_module(path: Path, root: Path) -> list[Finding]:
    """Audit public module-level functions and public class methods in one file."""
    tree = ast.parse(path.read_text(), filename=path)
    findings: list[Finding] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and public_name(node.name):
            if ast.get_docstring(node) is None:
                findings.append(
                    Finding(path, node.lineno, node.name, "missing docstring")
                )
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    public_name(member.name)
                ):
                    findings.extend(
                        audit_callable(path, member, f"{node.name}.{member.name}")
                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and public_name(
            node.name
        ):
            findings.extend(audit_callable(path, node, node.name))
    return findings


def python_files(directory: Path) -> tuple[Path, ...]:
    """Return Python sources recursively, excluding package initializers."""
    return tuple(
        path for path in sorted(directory.rglob("*.py")) if path.name != "__init__.py"
    )


def main() -> int:
    """Print an API-docstring coverage report for a Python source directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Python source directory to audit")
    arguments = parser.parse_args()
    directory = arguments.directory.resolve()
    if not directory.is_dir():
        parser.error(f"not a directory: {directory}")

    files = python_files(directory)
    findings = [finding for path in files for finding in audit_module(path, directory)]
    for finding in findings:
        relative = finding.path.relative_to(directory)
        print(f"{relative}:{finding.line}: {finding.symbol}: {finding.message}")
    print(f"Audited {len(files)} files; found {len(findings)} documentation gaps.")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
