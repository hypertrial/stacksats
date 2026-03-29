from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import scripts.release_wheel_smoke as release_wheel_smoke


class _TempDirContext:
    def __init__(self, root: Path):
        self._root = root

    def __enter__(self) -> str:
        self._root.mkdir(parents=True, exist_ok=True)
        return str(self._root)

    def __exit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        return False


def test_resolve_single_path_returns_existing_literal_path(tmp_path: Path) -> None:
    wheel_path = tmp_path / "dist" / "stacksats-1.0.0.whl"
    wheel_path.parent.mkdir(parents=True, exist_ok=True)
    wheel_path.write_text("wheel", encoding="utf-8")

    resolved = release_wheel_smoke._resolve_single_path(str(wheel_path), kind="wheel")

    assert resolved == wheel_path.resolve()


def test_resolve_single_path_expands_unique_glob_pattern(tmp_path: Path, monkeypatch) -> None:
    wheel_path = tmp_path / "dist" / "stacksats-1.0.0.whl"
    wheel_path.parent.mkdir(parents=True, exist_ok=True)
    wheel_path.write_text("wheel", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    resolved = release_wheel_smoke._resolve_single_path("dist/*.whl", kind="wheel")

    assert resolved == wheel_path.resolve()


@pytest.mark.parametrize(
    ("pattern", "expected_fragment", "create_matches"),
    [
        ("dist/*.whl", "found: \\[\\]", False),
        ("dist/*.whl", "Expected exactly one wheel", True),
    ],
)
def test_resolve_single_path_exits_when_glob_is_not_unique(
    tmp_path: Path,
    monkeypatch,
    pattern: str,
    expected_fragment: str,
    create_matches: bool,
) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    if create_matches:
        (dist_dir / "stacksats-a.whl").write_text("a", encoding="utf-8")
        (dist_dir / "stacksats-b.whl").write_text("b", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match=expected_fragment):
        release_wheel_smoke._resolve_single_path(pattern, kind="wheel")


@pytest.mark.parametrize("mode", ["base", "service", "all"])
def test_parse_args_accepts_supported_modes(monkeypatch, mode: str) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "release_wheel_smoke.py",
            "--wheel",
            "dist/stacksats.whl",
            "--constraints-file",
            "requirements.txt",
            "--mode",
            mode,
        ],
    )

    args = release_wheel_smoke._parse_args()

    assert args.wheel == "dist/stacksats.whl"
    assert args.constraints_file == "requirements.txt"
    assert args.mode == mode


def test_main_runs_only_base_smoke_for_base_mode(tmp_path: Path, monkeypatch, capsys) -> None:
    root = tmp_path / "release-root"
    wheel_path = tmp_path / "dist" / "stacksats.whl"
    constraints_file = tmp_path / "requirements.txt"
    wheel_path.parent.mkdir(parents=True, exist_ok=True)
    wheel_path.write_text("wheel", encoding="utf-8")
    constraints_file.write_text("constraints", encoding="utf-8")
    calls: list[tuple[str, Path, Path | None]] = []

    monkeypatch.setattr(
        release_wheel_smoke,
        "_parse_args",
        lambda: argparse.Namespace(
            wheel=str(wheel_path),
            constraints_file=str(constraints_file),
            mode="base",
        ),
    )
    monkeypatch.setattr(
        release_wheel_smoke,
        "_resolve_single_path",
        lambda pattern, *, kind: wheel_path.resolve()
        if kind == "wheel"
        else constraints_file.resolve(),
    )
    monkeypatch.setattr(release_wheel_smoke.shutil, "which", lambda executable: executable)
    monkeypatch.setattr(
        release_wheel_smoke.tempfile,
        "TemporaryDirectory",
        lambda prefix: _TempDirContext(root),
    )
    monkeypatch.setattr(
        release_wheel_smoke,
        "_base_smoke",
        lambda smoke_root, wheel: calls.append(("base", smoke_root, wheel)),
    )
    monkeypatch.setattr(
        release_wheel_smoke,
        "_viz_smoke",
        lambda smoke_root, wheel, constraints: calls.append(("viz", smoke_root, constraints)),
    )
    monkeypatch.setattr(
        release_wheel_smoke,
        "_service_smoke",
        lambda smoke_root, wheel: calls.append(("service", smoke_root, wheel)),
    )

    exit_code = release_wheel_smoke.main()

    assert exit_code == 0
    assert calls == [("base", root, wheel_path.resolve())]
    assert "Release wheel smoke passed for stacksats.whl (base)" in capsys.readouterr().out


def test_main_runs_only_service_smoke_for_service_mode(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    root = tmp_path / "release-root"
    wheel_path = tmp_path / "dist" / "stacksats.whl"
    constraints_file = tmp_path / "requirements.txt"
    wheel_path.parent.mkdir(parents=True, exist_ok=True)
    wheel_path.write_text("wheel", encoding="utf-8")
    constraints_file.write_text("constraints", encoding="utf-8")
    calls: list[tuple[str, Path, Path | None]] = []

    monkeypatch.setattr(
        release_wheel_smoke,
        "_parse_args",
        lambda: argparse.Namespace(
            wheel=str(wheel_path),
            constraints_file=str(constraints_file),
            mode="service",
        ),
    )
    monkeypatch.setattr(
        release_wheel_smoke,
        "_resolve_single_path",
        lambda pattern, *, kind: wheel_path.resolve()
        if kind == "wheel"
        else constraints_file.resolve(),
    )
    monkeypatch.setattr(release_wheel_smoke.shutil, "which", lambda executable: executable)
    monkeypatch.setattr(
        release_wheel_smoke.tempfile,
        "TemporaryDirectory",
        lambda prefix: _TempDirContext(root),
    )
    monkeypatch.setattr(
        release_wheel_smoke,
        "_base_smoke",
        lambda smoke_root, wheel: calls.append(("base", smoke_root, wheel)),
    )
    monkeypatch.setattr(
        release_wheel_smoke,
        "_viz_smoke",
        lambda smoke_root, wheel, constraints: calls.append(("viz", smoke_root, constraints)),
    )
    monkeypatch.setattr(
        release_wheel_smoke,
        "_service_smoke",
        lambda smoke_root, wheel: calls.append(("service", smoke_root, wheel)),
    )

    exit_code = release_wheel_smoke.main()

    assert exit_code == 0
    assert calls == [("service", root, wheel_path.resolve())]
    assert "Release wheel smoke passed for stacksats.whl (service)" in capsys.readouterr().out


def test_main_runs_base_viz_and_service_smokes_for_all_mode(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    root = tmp_path / "release-root"
    wheel_path = tmp_path / "dist" / "stacksats.whl"
    constraints_file = tmp_path / "requirements.txt"
    wheel_path.parent.mkdir(parents=True, exist_ok=True)
    wheel_path.write_text("wheel", encoding="utf-8")
    constraints_file.write_text("constraints", encoding="utf-8")
    calls: list[tuple[str, Path, Path, Path | None]] = []

    monkeypatch.setattr(
        release_wheel_smoke,
        "_parse_args",
        lambda: argparse.Namespace(
            wheel=str(wheel_path),
            constraints_file=str(constraints_file),
            mode="all",
        ),
    )
    monkeypatch.setattr(
        release_wheel_smoke,
        "_resolve_single_path",
        lambda pattern, *, kind: wheel_path.resolve()
        if kind == "wheel"
        else constraints_file.resolve(),
    )
    monkeypatch.setattr(release_wheel_smoke.shutil, "which", lambda executable: executable)
    monkeypatch.setattr(
        release_wheel_smoke.tempfile,
        "TemporaryDirectory",
        lambda prefix: _TempDirContext(root),
    )
    monkeypatch.setattr(
        release_wheel_smoke,
        "_base_smoke",
        lambda smoke_root, wheel: calls.append(("base", smoke_root, wheel, None)),
    )
    monkeypatch.setattr(
        release_wheel_smoke,
        "_viz_smoke",
        lambda smoke_root, wheel, constraints: calls.append(("viz", smoke_root, wheel, constraints)),
    )
    monkeypatch.setattr(
        release_wheel_smoke,
        "_service_smoke",
        lambda smoke_root, wheel: calls.append(("service", smoke_root, wheel, None)),
    )

    exit_code = release_wheel_smoke.main()

    assert exit_code == 0
    assert calls == [
        ("base", root, wheel_path.resolve(), None),
        ("viz", root, wheel_path.resolve(), constraints_file.resolve()),
        ("service", root, wheel_path.resolve(), None),
    ]
    assert "Release wheel smoke passed for stacksats.whl (all)" in capsys.readouterr().out


def test_main_exits_when_current_python_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    wheel_path = tmp_path / "dist" / "stacksats.whl"
    constraints_file = tmp_path / "requirements.txt"
    wheel_path.parent.mkdir(parents=True, exist_ok=True)
    wheel_path.write_text("wheel", encoding="utf-8")
    constraints_file.write_text("constraints", encoding="utf-8")

    monkeypatch.setattr(
        release_wheel_smoke,
        "_parse_args",
        lambda: argparse.Namespace(
            wheel=str(wheel_path),
            constraints_file=str(constraints_file),
            mode="base",
        ),
    )
    monkeypatch.setattr(
        release_wheel_smoke,
        "_resolve_single_path",
        lambda pattern, *, kind: wheel_path.resolve()
        if kind == "wheel"
        else constraints_file.resolve(),
    )
    monkeypatch.setattr(release_wheel_smoke.shutil, "which", lambda executable: None)

    with pytest.raises(SystemExit, match="Current Python executable is not available"):
        release_wheel_smoke.main()


def test_http_json_normalizes_response_headers_to_lowercase(monkeypatch) -> None:
    payload = {"status": "ok"}

    class _Headers:
        def items(self):
            return [("X-Request-ID", "Req-123"), ("Content-Type", "application/json")]

    class _Response:
        status = 200
        headers = _Headers()

        def read(self) -> bytes:
            return json.dumps(payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            del exc_type, exc, tb
            return False

    monkeypatch.setattr(
        release_wheel_smoke.urllib.request,
        "urlopen",
        lambda request, timeout: _Response(),
    )

    status_code, headers, parsed_payload = release_wheel_smoke._http_json(
        url="http://example.test/healthz"
    )

    assert status_code == 200
    assert headers == {
        "x-request-id": "Req-123",
        "content-type": "application/json",
    }
    assert parsed_payload == payload
