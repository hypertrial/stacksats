from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_sync_objects_schema_check_succeeds_without_manual_pythonpath() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "sync_objects_schema_docs.py"

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(script), "--check"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "schema sections are up to date" in result.stdout
