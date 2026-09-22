# Installation

Detectiv requires Python 3.12 or newer. Install the published package with pip:

```console
python -m pip install detectiv
```

The base package contains the Python experiment API. Install optional experiment
tracking and visualization support only when it is needed:

```console
python -m pip install 'detectiv[experiment]'
```

For an unreleased checkout, install directly from the repository with `uv sync`.
See [Development](development.md) for contributor setup.
