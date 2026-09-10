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
    uv run mypy src tests scenarios

test:
    uv run coverage erase
    uv run coverage run -m pytest
    uv run coverage report

docs:
    uv run mkdocs build --strict

build:
    uv build --no-sources

check: lint typecheck test docs build
