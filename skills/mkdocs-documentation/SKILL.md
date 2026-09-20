---
name: mkdocs-documentation
description: Use when creating or reviewing MkDocs pages, public Python docstrings, MkDocs navigation, or generated API references. Enforces maintainable user documentation and strict build validation.
license: MIT
compatibility: OpenCode and agents supporting the SKILL.md standard
metadata:
  version: "1"
  tags: "mkdocs, documentation, python, api-reference"
---

# MkDocs Documentation

Create documentation with one owner for each fact:

- Narrative MkDocs pages own workflows, concepts, setup, and runnable examples.
- Public Python docstrings own exact API contracts.
- Generated API references render public docstrings and signatures when enabled.
- Tests own executable behavior; documentation must not replace them.

## Public Docstrings

Document every public class, function, method, and configuration dataclass.

- Class docstrings state the type's responsibility and constructor configuration.
- Method docstrings state only that method's inputs, output, errors, side effects,
  ordering, laziness, or performance guarantees.
- Use `Args`, `Returns`, and `Raises` sections when they add contract information.
- Keep each method self-contained. Do not direct readers to sibling methods.
- Do not restate type annotations or obvious implementation steps.
- Do not add docstrings to private helpers to explain unclear behavior. Flag it
  as a refactoring recommendation; do not hide the problem with documentation.
- Add a brief private comment only for an irreducible external constraint, such
  as an upstream API quirk, platform behavior, or deliberate performance trade-off.

## Narrative Pages

- Teach one progressive user journey: prerequisite, setup, first successful
  command, expected result, then advanced use.
- Keep commands runnable and avoid undefined names or hardcoded user-specific
  identifiers.
- Link to authoritative upstream sources for third-party datasets and tools.
- Use relative links for local pages and verify them with a strict build.
- Keep generated data, secrets, credentials, and large downloadable datasets out
  of the repository.

## API References

- Link narrative concepts to public API references when they improve navigation.
- Do not add a generated API plugin or configuration without first checking the
  existing MkDocs setup and confirming the desired public surface.
- Never rely on narrative documentation to describe every argument; the API
  reference is the source of truth for signatures and contract details.

## Validation

Run after documentation changes:

```shell
uv run mkdocs build --strict
```

If Python docstrings change, run the focused tests and repository quality checks
required by the affected public API.
