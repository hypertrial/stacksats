from __future__ import annotations

import tomllib
from pathlib import Path
import re


WORKFLOW_FILES = (
    ".github/workflows/coverage-report.yml",
    ".github/workflows/docs-check.yml",
    ".github/workflows/docs-pages.yml",
    ".github/workflows/example-commands-smoke.yml",
    ".github/workflows/package-check-pr.yml",
    ".github/workflows/package-check.yml",
    ".github/workflows/release-gate.yml",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read(rel: str) -> str:
    return (_repo_root() / rel).read_text(encoding="utf-8")


def _workflow_texts() -> dict[str, str]:
    return {path: _read(path) for path in WORKFLOW_FILES}


def test_github_workflows_use_current_action_versions() -> None:
    workflow_texts = _workflow_texts()
    helper_text = _read(".github/actions/setup-python-project/action.yml")
    combined = "\n".join([*workflow_texts.values(), helper_text])

    assert "actions/checkout@v4" not in combined
    assert "actions/setup-python@v5" not in combined
    assert "actions/upload-artifact@v4" not in combined
    assert "actions/download-artifact@v4" not in combined
    for workflow_path, workflow_text in workflow_texts.items():
        assert "actions/checkout@v6" in workflow_text, workflow_path
    assert "actions/setup-python@v6" in helper_text
    assert "actions/setup-python@v6" in workflow_texts[".github/workflows/package-check.yml"]
    assert "actions/setup-python@v6" in workflow_texts[".github/workflows/package-check-pr.yml"]
    assert "actions/setup-python@v6" in workflow_texts[".github/workflows/release-gate.yml"]
    assert "actions/upload-artifact@v6" in workflow_texts[".github/workflows/coverage-report.yml"]
    assert "actions/upload-artifact@v6" in workflow_texts[".github/workflows/package-check.yml"]
    assert "actions/upload-artifact@v6" in workflow_texts[".github/workflows/package-check-pr.yml"]
    assert "actions/upload-artifact@v6" in workflow_texts[".github/workflows/release-gate.yml"]
    assert "actions/download-artifact@v8" in workflow_texts[".github/workflows/package-check.yml"]
    assert "actions/download-artifact@v8" in workflow_texts[".github/workflows/package-check-pr.yml"]
    assert "actions/download-artifact@v8" in workflow_texts[".github/workflows/release-gate.yml"]


def test_reusable_python_setup_action_is_wired_into_expected_workflows() -> None:
    workflow_texts = _workflow_texts()
    helper_action = _read(".github/actions/setup-python-project/action.yml")

    assert "name: setup-python-project" in helper_action
    assert "cache: pip" in helper_action
    for workflow_path, workflow_text in workflow_texts.items():
        assert "uses: ./.github/actions/setup-python-project" in workflow_text
        assert "constraints-file: " in workflow_text, workflow_path


def test_ci_workflow_contracts_keep_critical_gates() -> None:
    workflow_texts = _workflow_texts()
    coverage_report = workflow_texts[".github/workflows/coverage-report.yml"]
    package_check = workflow_texts[".github/workflows/package-check.yml"]
    package_check_pr = workflow_texts[".github/workflows/package-check-pr.yml"]
    release_gate = workflow_texts[".github/workflows/release-gate.yml"]
    docs_pages = workflow_texts[".github/workflows/docs-pages.yml"]
    docs_check = workflow_texts[".github/workflows/docs-check.yml"]
    example_commands_smoke = workflow_texts[".github/workflows/example-commands-smoke.yml"]

    assert "name: package-check" in package_check
    assert 'pytest ${{ matrix.test_target }} -v -m "${{ matrix.marker }}"' in package_check
    assert "python -m mkdocs build --strict" in package_check
    assert "python -m build" in package_check

    assert "name: package-check-pr" in package_check_pr
    assert "timing-budget:" in package_check_pr
    assert "PR runtime budget exceeded" in package_check_pr
    assert "python -m build" in package_check_pr

    assert "name: release-gate" in release_gate
    assert "bash scripts/check_coverage.sh" in release_gate
    assert "python scripts/check_release_docs_sync.py" in release_gate
    assert "python scripts/release_wheel_smoke.py" in release_gate
    assert "--mode all" in release_gate

    assert "name: docs-pages" in docs_pages
    assert "python -m mkdocs build --strict" in docs_pages
    assert 'FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: "true"' in docs_pages
    assert "actions/configure-pages@v5" in docs_pages
    assert "actions/upload-pages-artifact@v4" not in docs_pages
    assert "tar --directory site -cf \"$RUNNER_TEMP/github-pages.tar\" ." in docs_pages
    assert "gzip -f \"$RUNNER_TEMP/github-pages.tar\"" in docs_pages
    assert "actions/upload-artifact@v6" in docs_pages
    assert "name: github-pages" in docs_pages
    assert "path: ${{ runner.temp }}/github-pages.tar.gz" in docs_pages
    assert "actions/deploy-pages@v4" in docs_pages
    assert 'timeout: "1800000"' in docs_pages
    assert "name: docs-check" in docs_check
    assert "python scripts/check_release_docs_sync.py" in docs_check
    assert "python scripts/check_docs_ux.py" in docs_check
    assert "lycheeverse/lychee-action@v2" in docs_check

    assert "name: example-commands-smoke" in example_commands_smoke
    assert "pytest tests/unit/core/test_distribution_wheel_smoke.py -q" in example_commands_smoke
    assert "python scripts/test_example_commands.py" in example_commands_smoke

    assert "name: coverage-report" in coverage_report
    assert "bash scripts/check_coverage.sh" in coverage_report
    assert "name: coverage-xml" in coverage_report
    assert "path: coverage.xml" in coverage_report
    assert "service_wheel_smoke:" in package_check
    assert "name: package-check-dist" in package_check
    assert "--mode service" in package_check


def test_docs_home_contract_is_explicit() -> None:
    docs_home = _read("docs/index.md")
    assert re.search(r"^##\s+Start in 2 Clicks$", docs_home, flags=re.MULTILINE)
    assert re.search(r"^##\s+Choose Your Path$", docs_home, flags=re.MULTILINE)
    assert re.search(r"^##\s+Feedback$", docs_home, flags=re.MULTILINE)
    assert "[Quickstart](start/quickstart.md)" in docs_home
    assert "[Task Hub](tasks.md)" in docs_home


def test_coverage_contract_is_explicit_and_package_only() -> None:
    repo_root = _repo_root()
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    coverage_run = pyproject["tool"]["coverage"]["run"]

    assert coverage_run["source"] == ["stacksats"]
    assert coverage_run["branch"] is True

    script = (repo_root / "scripts" / "check_coverage.sh").read_text(encoding="utf-8")
    assert 'COVERAGE_FAIL_UNDER="${COVERAGE_FAIL_UNDER:-100}"' in script
    assert '--cov=stacksats \\' in script
    assert '--cov-branch \\' in script
    assert '-m "not performance"' in script

    release_gate = (repo_root / ".github" / "workflows" / "release-gate.yml").read_text(
        encoding="utf-8"
    )
    assert "coverage:" in release_gate
    assert "bash scripts/check_coverage.sh" in release_gate
