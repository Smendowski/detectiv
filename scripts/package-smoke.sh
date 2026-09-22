#!/usr/bin/env bash
set -euo pipefail

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

uv venv "$tmpdir/venv"
uv pip install --python "$tmpdir/venv/bin/python" dist/*.whl
mkdir "$tmpdir/samples"
cp samples/01_synthetic_reconstruction.py "$tmpdir/samples/"
(cd "$tmpdir" && "$tmpdir/venv/bin/python" samples/01_synthetic_reconstruction.py)
