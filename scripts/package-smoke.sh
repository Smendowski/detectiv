#!/usr/bin/env bash
set -euo pipefail

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

uv venv "$tmpdir/venv"
uv pip install --python "$tmpdir/venv/bin/python" dist/*.whl
cp examples/first_experiment.py "$tmpdir/first_experiment.py"
(cd "$tmpdir" && "$tmpdir/venv/bin/python" first_experiment.py)
