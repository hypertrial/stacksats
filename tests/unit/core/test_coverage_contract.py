from __future__ import annotations

import tomllib
from pathlib import Path


def test_coverage_contract_is_explicit_and_package_only() -> None:
    repo_root = Path(__file__).resolve().parents[3]
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
