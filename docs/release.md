---
title: Release Guide
description: Manual-first release process for StackSats package and documentation workflows.
---

# Release Guide

This guide is manual-first. It covers token-based PyPI releases.

For `1.0.0` and later, treat `release-gate.yml` as the only release-signoff workflow. Fast PR and `main` workflows are confidence lanes, not approval lanes.

## Release Policy

- Use SemVer: `MAJOR.MINOR.PATCH`.
- Package version is generated automatically from git tags via `setuptools-scm`.
- Use annotated git tags in the form `vX.Y.Z`.
- Tag and package version must match exactly (for example, tag `v0.7.1` produces package version `0.7.1`).
- Never reuse a version number after it has been uploaded to PyPI.
- Do not publish artifacts built before the release tag exists. Build the upload artifacts from the tagged `vX.Y.Z` commit context.

## One-Time Setup

### Accounts and project names

1. Create/verify account on:
   - PyPI: `https://pypi.org/`
2. Ensure package name `stacksats` is available/owned by this project.

### Local tooling

Use Python 3.11+ and install packaging tools:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -c requirements/constraints-maintainer.txt -e ".[dev,all]"
venv/bin/python -m pip install -c requirements/constraints-maintainer.txt --upgrade build twine
```

### Local token handling (default)

Use PyPI API tokens locally only. Do not commit them.

Example with environment variables:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD="<pypi-token>"
```

If your local workflow uses `PYPI_API_KEY` in `.env`, map it at runtime:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD="${PYPI_API_KEY}"
```

## Manual Release Checklist

### 1) Prepare release branch/PR

1. Sanity-check that `README.md` deep-links into docs instead of duplicating long CLI matrices (see `docs/docs_ownership.md`).
2. Update `CHANGELOG.md` (required):
   - Add user-visible changes under `## [Unreleased]`.
   - Before tagging, move unreleased entries into the new `vX.Y.Z` section with the release date.
   - Start a fresh `## [Unreleased]` section for follow-up work.
3. Update `docs/whats-new.md` so the latest released section still matches the newest changelog release.
4. Ensure CI/tests are green.
5. Merge to `main`.
6. Create a release branch such as `release/1.0.0` for final pre-tag verification.

### 1a) Verify GitHub protections

Before cutting a `1.x` release, verify repository settings are aligned with the release process:

- `main` requires pull requests.
- `release/*` branches require the `release-gate` workflow.
- Release tags require the `release-gate` workflow before publish.
- Normal maintainer bypass is disabled unless there is an explicit emergency policy exception.

### 2) Local preflight checks

From repository root:

```bash
bash scripts/release_check.sh
```

This is a release-preflight command. It runs lint, docs checks, generated-doc sync checks, the full non-performance release suite, a preflight package build, and `twine check`.
It also runs the BRK source-contract guard (`scripts/check_no_coinmetrics_refs.py`).
The preflight build only verifies buildability; rebuild release artifacts after the release tag is created.

If your environment has constrained SSL trust roots, `scripts/release_check.sh` will:
- Fall back to already-installed local `build`/`twine` when tool refresh via pip fails.
- Retry package build with `python -m build --no-isolation` only when isolated-build failure looks SSL-related.
- Still fail normally for non-SSL build errors.

### 3) Push release branch and verify release gate

Push the release candidate to a release branch and wait for `release-gate.yml` to pass:

```bash
git push origin release/X.Y.Z
```

The release gate must pass on the release branch before you create the annotated tag. It enforces:

- Linux quality checks, strict docs build, full non-performance tests, package build, and metadata validation
- isolated built-wheel smoke validation
- macOS Python 3.11 supported-platform smoke validation

### 4) Build artifacts

Create and push the release tag before building the publishable artifacts:

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

The tag is the source of truth for the version. No manual version bump is required.

Then build from the tagged commit context:

```bash
rm -rf dist/ build/ .eggs/ *.egg-info
venv/bin/python -m build
venv/bin/python -m twine check dist/*
```

### 5) Publish to PyPI (default)

```bash
bash scripts/publish_pypi_manual.sh
```

### 6) Post-release verification

- Verify package page on PyPI.
- Install from PyPI in a fresh virtual environment.
- Verify the stable CLI entry point:
  - `stacksats`
- Confirm the required fresh-install smoke covers:
  - `stacksats demo backtest`
  - `stacksats-plot-mvrv` in an environment with `viz` installed
- Optional manual helper verification when relevant:
  - `stacksats-plot-weights` in an environment with both `viz` and `deploy` installed plus a valid `DATABASE_URL`

## CI Workflow Notes

- Verification lanes at a glance:

| Workflow | Role | What it proves |
| --- | --- | --- |
| `package-check-pr.yml` | fast PR confidence | quick lint/docs/tests/package confidence before merge |
| `package-check.yml` | main confidence | ongoing confidence on `main`, not release approval |
| `docs-check.yml` | docs quality | markdown, spelling, links, docs sync, docs UX, strict docs build |
| `docs-pages.yml` | docs publish | strict docs build plus GitHub Pages deploy from `main` |
| `example-commands-smoke.yml` | scheduled/manual command smoke | docs-example regression lane via `scripts/test_example_commands.py` plus inherited-environment wheel regression |
| `coverage-report.yml` | scheduled/manual maintenance coverage | branch-aware coverage visibility via `scripts/check_coverage.sh` |
| `release-gate.yml` | release signoff | release-grade quality/docs/tests/coverage/package validation plus isolated wheel smoke via `scripts/release_wheel_smoke.py` |

- `release-gate.yml` is the release-grade blocking workflow for `release/*` branches and `v*` tags.
- `release-gate.yml` now validates the built wheel in isolated virtual environments. It does not rely on inherited site-packages.
- `release-gate.yml` covers the stable `stacksats` CLI path plus the optional `stacksats-plot-mvrv` helper; it does not attempt database-backed `stacksats-plot-weights` smoke in CI.
- Pull requests run fast confidence checks (`package-check-pr.yml`) and docs checks. They are intentionally faster than the release gate and are not release sign-off.
- Pushes to `main` run `package-check.yml` for ongoing confidence, not release approval.
- Full non-performance branch-aware coverage also runs in `release-gate.yml`; `coverage-report.yml` remains scheduled/manual maintenance visibility.
- Coverage fail-under is `100%` line and branch coverage for `stacksats/` and should not be lowered in routine maintenance PRs.
- CLI docs examples are smoke-tested in scheduled/manual `example-commands-smoke.yml` via `scripts/test_example_commands.py`.
- PyPI publishing is manual only via `scripts/publish_pypi_manual.sh`.
- Pull requests also run docs quality checks (`docs-check.yml`):
  - markdown lint across all tracked `.md` files
  - spelling checks across all tracked `.md` files
  - link checks across all tracked `.md` files
  - release docs sync check
  - docs reference checks
  - docs UX structure checks
  - strict docs build
- Both `docs-check.yml` and `docs-pages.yml` install with `requirements/constraints-maintainer.txt` and `.[dev,all]` so docs environments stay aligned with release-grade installs.
- Pushes to `main` publish docs to GitHub Pages via `docs-pages.yml`.
- `example-commands-smoke.yml` keeps a lighter inherited-environment wheel regression test for maintainers, but it is not the release artifact gate.
- `release-gate.yml` is still the only release-signoff workflow; none of the faster confidence or scheduled/manual lanes replace it.

## GitHub Pages Source

Configure Pages source to `GitHub Actions`.

## End-to-End Validation Runbook

Use this sequence after workflows are merged:

1. Open a PR with release notes/changelog updates and verify `package-check-pr.yml` passes.
2. Run `bash scripts/release_check.sh` on the release candidate commit.
3. Push the release candidate to a `release/*` branch and verify `release-gate.yml` passes.
4. Create and push annotated tag `vX.Y.Z`.
5. Verify the tag-triggered `release-gate.yml` run passes.
6. Rebuild from the tagged commit context and run `twine check dist/*`.
7. Run `bash scripts/publish_pypi_manual.sh`.
8. Verify package page on PyPI.
9. Install from PyPI in a fresh virtual environment and run command smoke tests.

Expected results:

- Manual publish succeeds with local `PYPI_API_KEY`.
- Tagged build and `twine check` pass before upload.
- Release branch and tagged release both pass `release-gate.yml` before publish.

## Operational Notes

- Do not commit PyPI tokens or store them in repository files.
- Keep `scripts/publish_pypi_manual.sh` as the default release path.
- If a release fails after version/tag creation, bump to the next version and retry; do not overwrite versions.
- If `scripts/release_check.sh` changes, update this page and `CONTRIBUTING.md` in the same PR.
- If GitHub branch/tag protection settings change, update this page in the same PR that changes the policy.
- Keep contributor and policy docs current:
  - `CONTRIBUTING.md`
  - `SECURITY.md`
  - `CODE_OF_CONDUCT.md`
