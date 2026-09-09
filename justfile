set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

bootstrap:
    uv sync --locked --all-groups
    uv run prek install

format:
    uv run ruff check --fix .
    uv run ruff format .

lint:
    uv run ruff check .
    uv run ruff format --check .

typecheck:
    uv run mypy src tests

test:
    uv run coverage erase
    uv run coverage run -m pytest
    uv run coverage report

docs:
    uv run sphinx-build --fail-on-warning --keep-going -b html docs docs/_build/html

build:
    uv build --no-sources

check: lint typecheck test docs build
