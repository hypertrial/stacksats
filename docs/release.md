---
title: Release Guide
description: Manual and automated release process for StackSats package and documentation workflows.
---

# Release Guide

This guide covers both:

- Manual PyPI releases (immediate baseline).
- Automated releases using Trusted Publishing with GitHub Actions OIDC.

## Release Policy

- Use SemVer: `MAJOR.MINOR.PATCH`.
- Package version is generated automatically from git tags via `setuptools-scm`.
- Use annotated git tags in the form `vX.Y.Z`.
- Tag and package version must match exactly (for example, tag `v0.1.1` produces package version `0.1.1`).
- Never reuse a version number after it has been uploaded to PyPI.

## One-Time Setup

### Accounts and project names

1. Create/verify account on:
   - PyPI: `https://pypi.org/`
2. Ensure package name `stacksats` is available/owned by this project.

### Local tooling

Use Python 3.11+ and install packaging tools:

```bash
python -m pip install --upgrade pip
python -m pip install --upgrade build twine
```

### Local token handling (manual fallback)

For manual uploads, use PyPI API tokens locally only. Do not commit them.

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

This should run lint, tests, build, and `twine check`.

If your environment has constrained SSL trust roots, `scripts/release_check.sh` will:
- Fall back to already-installed local `build`/`twine` when tool refresh via pip fails.
- Retry package build with `python -m build --no-isolation` only when isolated-build failure looks SSL-related.
- Still fail normally for non-SSL build errors.

### 3) Build artifacts

```bash
rm -rf dist/ build/ .eggs/ *.egg-info
python -m build
python -m twine check dist/*
```

### 4) Publish to PyPI

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD="<pypi-token>"
python -m twine upload dist/*
```

### 5) Tag release

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

The tag is the source of truth for the version. No manual version bump is required.

### 6) Post-release verification

- Verify package page on PyPI.
- Install from PyPI in a fresh virtual environment.
- Verify entry points:
  - `stacksats`
  - `stacksats-plot-mvrv`
  - `stacksats-plot-weights`

## Trusted Publishing (OIDC) Setup

Trusted Publishing removes the need for stored PyPI secrets in GitHub.

### Workflow files used by this repository

- `.github/workflows/package-check.yml` (push to `main`)
- `.github/workflows/package-check-pr.yml` (pull requests)
- `.github/workflows/publish-pypi.yml`
- `.github/workflows/docs-check.yml`
- `.github/workflows/docs-pages.yml`

### Configure Trusted Publisher on PyPI

In PyPI project settings, add a Trusted Publisher with:

- Owner: `hypertrial`
- Repository: `stacksats`
- Workflow filename: `publish-pypi.yml`
- Environment: leave blank unless the workflow job is explicitly bound to a GitHub environment.

### Optional GitHub environment protections

If you bind the publish job to a GitHub environment (for example `pypi`), the PyPI trusted publisher must include the exact same environment name.

## Automated Release Flow

- Pull requests run packaging checks (`package-check-pr.yml`).
- Pushes to `main` run packaging checks (`package-check.yml`).
- Pushes of tags matching `v*` publish to PyPI (`publish-pypi.yml`).
- PyPI publish job validates tag/version consistency before upload.
- Pull requests also run docs quality checks (`docs-check.yml`):
  - markdown lint
  - spelling checks
  - link checks
  - docs reference checks
  - strict docs build
- Pushes to `main` publish docs to GitHub Pages via `docs-pages.yml`.

## GitHub Pages Source

Configure Pages source to `GitHub Actions`.

## End-to-End Validation Runbook

Use this sequence after workflows are merged:

1. Open a PR with release notes/changelog updates and verify `package-check-pr.yml` passes.
2. Create and push annotated tag `vX.Y.Z`.
3. Verify `publish-pypi.yml` succeeds.
4. Install from PyPI in a fresh virtual environment and run command smoke tests.

Expected results:

- No PyPI API tokens are configured in GitHub repository secrets.
- PyPI publishes only from `v*` tags.
- Tag/version mismatch fails before publish.

## Operational Notes

- Do not store PyPI tokens in GitHub Secrets when using OIDC Trusted Publishing.
- Keep manual twine upload steps documented for emergency fallback only.
- If a release fails after version/tag creation, bump to the next version and retry; do not overwrite versions.
- Keep contributor and policy docs current:
  - `CONTRIBUTING.md`
  - `SECURITY.md`
  - `CODE_OF_CONDUCT.md`
