# Contributing

## Setup

Install [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/), then run:

```console
just bootstrap
```

## Development workflow

Format changes and run the complete validation suite before submitting them:

```console
just format
just check
```

Add tests for observable behavior and bug fixes. Public interfaces must include type annotations and documentation.
