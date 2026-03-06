---
title: Release Guide
description: Manual-first release process for StackSats package and documentation workflows.
---

# Release Guide

This guide is manual-first. It covers token-based PyPI releases.

## Release Policy

- Use SemVer: `MAJOR.MINOR.PATCH`.
- Package version is generated automatically from git tags via `setuptools-scm`.
- Use annotated git tags in the form `vX.Y.Z`.
- Tag and package version must match exactly (for example, tag `v0.1.1` produces package version `0.1.1`).
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
pip install -e ".[dev]"
python -m pip install --upgrade build twine
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

1. Update `CHANGELOG.md` (required):
   - Add user-visible changes under `## [Unreleased]`.
   - Before tagging, move unreleased entries into the new `vX.Y.Z` section with the release date.
   - Start a fresh `## [Unreleased]` section for follow-up work.
2. Ensure CI/tests are green.
3. Merge to `main`.

### 2) Local preflight checks

From repository root:

```bash
bash scripts/release_check.sh
```

This is a release-preflight command. It runs lint, docs checks, the full non-performance test suite, a preflight package build, and `twine check`.
The preflight build only verifies buildability; rebuild release artifacts after the release tag is created.

If your environment has constrained SSL trust roots, `scripts/release_check.sh` will:
- Fall back to already-installed local `build`/`twine` when tool refresh via pip fails.
- Retry package build with `python -m build --no-isolation` only when isolated-build failure looks SSL-related.
- Still fail normally for non-SSL build errors.

### 3) Build artifacts

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

### 4) Publish to PyPI (default)

```bash
bash scripts/publish_pypi_manual.sh
```

### 5) Post-release verification

- Verify package page on PyPI.
- Install from PyPI in a fresh virtual environment.
- Verify entry points:
  - `stacksats`
  - `stacksats-plot-mvrv`
  - `stacksats-plot-weights`

## CI Workflow Notes

- Pull requests run fast packaging checks (`package-check-pr.yml`) and docs checks.
- Pushes to `main` that match workflow scope run packaging and docs checks (`package-check.yml`) without the heavy coverage gate.
- Full non-performance coverage runs in the scheduled/manual `coverage-report.yml` workflow.
- Coverage fail-under is ratcheted upward over time and should not be lowered in routine maintenance PRs.
- CLI docs examples are smoke-tested in scheduled/manual `example-commands-smoke.yml`.
- PyPI publishing is manual only via `scripts/publish_pypi_manual.sh`.
- Pull requests also run docs quality checks (`docs-check.yml`):
  - markdown lint across all tracked `.md` files
  - spelling checks across all tracked `.md` files
  - link checks across all tracked `.md` files
  - release docs sync check
  - docs reference checks
  - docs UX structure checks
  - strict docs build
- Pushes to `main` publish docs to GitHub Pages via `docs-pages.yml`.

## GitHub Pages Source

Configure Pages source to `GitHub Actions`.

## End-to-End Validation Runbook

Use this sequence after workflows are merged:

1. Open a PR with release notes/changelog updates and verify `package-check-pr.yml` passes.
2. Run `bash scripts/release_check.sh` on the release candidate commit.
3. Create and push annotated tag `vX.Y.Z`.
4. Rebuild from the tagged commit context and run `twine check dist/*`.
5. Run `bash scripts/publish_pypi_manual.sh`.
6. Verify package page on PyPI.
7. Install from PyPI in a fresh virtual environment and run command smoke tests.

Expected results:

- Manual publish succeeds with local `PYPI_API_KEY`.
- Tagged build and `twine check` pass before upload.

## Operational Notes

- Do not commit PyPI tokens or store them in repository files.
- Keep `scripts/publish_pypi_manual.sh` as the default release path.
- If a release fails after version/tag creation, bump to the next version and retry; do not overwrite versions.
- If `scripts/release_check.sh` changes, update this page and `CONTRIBUTING.md` in the same PR.
- Keep contributor and policy docs current:
  - `CONTRIBUTING.md`
  - `SECURITY.md`
  - `CODE_OF_CONDUCT.md`
