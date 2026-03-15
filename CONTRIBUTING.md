# Contributing to StackSats

Thanks for your interest in contributing.

## Development setup

From the repository root:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pip install pre-commit
venv/bin/python -m pre_commit install -t pre-commit
```

Optional deploy extras:

```bash
pip install -e ".[deploy]"
```

## Local quality checks

Fast local checks:

```bash
venv/bin/python -m ruff check .
venv/bin/python -m pytest -q
bash scripts/check_docs_refs.sh
venv/bin/python scripts/check_docs_ux.py
venv/bin/python scripts/check_release_docs_sync.py
venv/bin/python scripts/sync_objects_schema_docs.py --check
venv/bin/python scripts/check_no_coinmetrics_refs.py
venv/bin/python -m mkdocs build --strict
```

Heavy test tiers when you need them explicitly:

```bash
venv/bin/python -m pytest -m "slow or integration or performance" -q
bash scripts/check_coverage.sh
bash scripts/clean_local.sh
```

Release-grade verification:

```bash
bash scripts/release_check.sh
```

Hook behavior:
- `pre-commit` (every commit): YAML sanity, whitespace fixes, `ruff`, docs reference checks, schema sync check.

Use `bash scripts/release_check.sh` for release prep only. It intentionally runs the full non-performance suite in addition to build/docs checks.
Current source contract is BRK parquet–only; keep runtime, docs, and tests aligned with canonical BRK parquet naming and providers.

## Contribution workflow

1. Create a feature branch from `main`.
2. Make focused changes with tests where appropriate.
3. Update docs and `CHANGELOG.md` (`Unreleased` section) for user-visible changes.
4. Open a pull request with a clear description and test evidence.

## Pull request expectations

- Keep behavior changes explicit and documented.
- Prefer small, reviewable PRs over large mixed changes.
- Include test coverage for fixes/features when practical.
- Full non-performance coverage (`bash scripts/check_coverage.sh`) runs in the scheduled/manual
  `coverage-report.yml` workflow and remains recommended before release cuts.
- Coverage fail-under is ratcheted upward over time; do not lower the floor in routine cleanup PRs.
- Avoid committing secrets or environment files.
- Follow docs ownership and update-trigger rules in `docs/docs_ownership.md`.
- If you change release tooling, docs test tiers, or markdown workflow scope, update `docs/release.md`, `README.md`, and `docs/docs_ownership.md` in the same PR.

## Release notes policy

If your change affects users, APIs, CLI behavior, packaging, or docs surfaced on PyPI, add an entry to `CHANGELOG.md`.

## Questions

Open a GitHub issue for questions, bugs, or feature requests.
