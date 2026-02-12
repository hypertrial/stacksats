# Release Guide

This guide covers both:

- Manual PyPI releases (immediate baseline).
- Automated releases using Trusted Publishing with GitHub Actions OIDC.

## Release Policy

- Use SemVer: `MAJOR.MINOR.PATCH`.
- Package version is generated automatically from git tags via `setuptools-scm`.
- Use annotated git tags in the form `vX.Y.Z`.
- Tag and package version must match exactly (for example, tag `v0.1.1` produces package version `0.1.1`).
- Never reuse a version number after it has been uploaded to TestPyPI or PyPI.

## One-Time Setup

### Accounts and project names

1. Create/verify accounts on both:
   - PyPI: `https://pypi.org/`
   - TestPyPI: `https://test.pypi.org/`
2. Ensure package name `stacksats` is available/owned by this project.

### Local tooling

Use Python 3.11+ and install packaging tools:

```bash
python -m pip install --upgrade pip
python -m pip install --upgrade build twine
```

### Local token handling (manual fallback)

For manual uploads, use API tokens locally only. Do not commit them.

Example with environment variables:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD="<pypi-or-testpypi-token>"
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

### 3) Build artifacts

```bash
rm -rf dist/ build/ .eggs/ *.egg-info
python -m build
python -m twine check dist/*
```

### 4) Publish to TestPyPI

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD="<testpypi-token>"
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

### 5) Smoke test from TestPyPI

```bash
python -m venv /tmp/stacksats-testpypi
source /tmp/stacksats-testpypi/bin/activate
python -m pip install --upgrade pip
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple stacksats
stacksats-backtest --help
stacksats-validate --help
```

### 6) Publish to PyPI

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD="<pypi-token>"
python -m twine upload dist/*
```

### 7) Tag release

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

The tag is the source of truth for the version. No manual version bump is required.

### 8) Post-release verification

- Verify package page on PyPI.
- Install from PyPI in a fresh virtual environment.
- Verify entry points:
  - `stacksats-backtest`
  - `stacksats-export`
  - `stacksats-plot-mvrv`
  - `stacksats-plot-weights`
  - `stacksats-validate`

## Trusted Publishing (OIDC) Setup

Trusted Publishing removes the need for stored PyPI secrets in GitHub.

### Workflow files used by this repository

- `.github/workflows/package-check.yml`
- `.github/workflows/publish-testpypi.yml`
- `.github/workflows/publish-pypi.yml`

### Configure Trusted Publisher on TestPyPI

In TestPyPI project settings, add a Trusted Publisher with:

- Owner: `hypertrial`
- Repository: `stacksats`
- Workflow filename: `publish-testpypi.yml`
- Environment (recommended): `testpypi`

### Configure Trusted Publisher on PyPI

In PyPI project settings, add a Trusted Publisher with:

- Owner: `hypertrial`
- Repository: `stacksats`
- Workflow filename: `publish-pypi.yml`
- Environment (recommended): `pypi`

### Recommended GitHub environment protections

- `testpypi`: optional reviewers.
- `pypi`: required reviewer(s), protected branch/tag rules.

## Automated Release Flow

- Pull requests run packaging checks (`package-check.yml`).
- Pushes to `main` publish to TestPyPI (`publish-testpypi.yml`).
- Pushes of tags matching `v*` publish to PyPI (`publish-pypi.yml`).
- PyPI publish job validates tag/version consistency before upload.

## End-to-End Validation Runbook

Use this sequence after workflows are merged:

1. Open a PR with a version bump and verify `package-check.yml` passes.
2. Merge PR to `main` and verify `publish-testpypi.yml` succeeds.
3. Install from TestPyPI and run command smoke tests.
4. Create and push annotated tag `vX.Y.Z`.
5. Verify `publish-pypi.yml` succeeds.
6. Install from PyPI in a fresh virtual environment and run command smoke tests.

Expected results:

- No PyPI/TestPyPI API tokens are configured in GitHub repository secrets.
- TestPyPI publishes only from `main`.
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
