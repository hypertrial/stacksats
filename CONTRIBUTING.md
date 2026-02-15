# Contributing to StackSats

Thanks for your interest in contributing.

## Development setup

From the repository root:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Optional deploy extras:

```bash
pip install -e ".[deploy]"
```

## Local quality checks

Run these before opening a pull request:

```bash
ruff check .
bash scripts/check_docs_refs.sh
python scripts/sync_objects_schema_docs.py --check
mkdocs build --strict
pytest tests/ -v
python -m build
python -m twine check dist/*
```

You can also run the project helper:

```bash
bash scripts/release_check.sh
```

## Contribution workflow

1. Create a feature branch from `main`.
2. Make focused changes with tests where appropriate.
3. Update docs and `CHANGELOG.md` (`Unreleased` section) for user-visible changes.
4. Open a pull request with a clear description and test evidence.

## Pull request expectations

- Keep behavior changes explicit and documented.
- Prefer small, reviewable PRs over large mixed changes.
- Include test coverage for fixes/features when practical.
- Avoid committing secrets or environment files.
- Follow docs ownership and update-trigger rules in `docs/docs_ownership.md`.

## Release notes policy

If your change affects users, APIs, CLI behavior, packaging, or docs surfaced on PyPI, add an entry to `CHANGELOG.md`.

## Questions

Open a GitHub issue for questions, bugs, or feature requests.
