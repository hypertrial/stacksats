from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import sys
import urllib.error

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_script_module(name: str):
    root = _repo_root()
    path = root / "scripts" / f"{name}.py"
    spec = spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stamp_docs_pages_marker = _load_script_module("stamp_docs_pages_marker")
check_docs_pages_canary = _load_script_module("check_docs_pages_canary")


def test_stamp_html_injects_marker_block_before_head_close() -> None:
    html = "<html><head><title>Docs</title></head><body>ok</body></html>"

    stamped = stamp_docs_pages_marker.stamp_html(
        html,
        commit="abc123",
        built_at="2026-03-30T15:00:00Z",
    )

    assert stamp_docs_pages_marker.MARKER_BLOCK_START in stamped
    assert '<meta name="stacksats-docs-commit" content="abc123">' in stamped
    assert '<meta name="stacksats-docs-built-at" content="2026-03-30T15:00:00Z">' in stamped
    assert stamped.index(stamp_docs_pages_marker.MARKER_BLOCK_START) < stamped.index("</head>")


def test_stamp_html_replaces_existing_marker_block_without_duplication() -> None:
    html = (
        "<html><head><title>Docs</title>\n"
        f"{stamp_docs_pages_marker.MARKER_BLOCK_START}\n"
        '<meta name="stacksats-docs-commit" content="old">\n'
        '<meta name="stacksats-docs-built-at" content="old-time">\n'
        f"{stamp_docs_pages_marker.MARKER_BLOCK_END}\n"
        "</head><body>ok</body></html>"
    )

    stamped = stamp_docs_pages_marker.stamp_html(
        html,
        commit="newsha",
        built_at="2026-03-30T16:00:00Z",
    )

    assert stamped.count(stamp_docs_pages_marker.MARKER_BLOCK_START) == 1
    assert 'content="newsha"' in stamped
    assert 'content="2026-03-30T16:00:00Z"' in stamped
    assert 'content="old"' not in stamped
    assert stamped.index(stamp_docs_pages_marker.MARKER_BLOCK_START) < stamped.index("</head>")
    assert stamped.index("</head>") < stamped.index("<body>")


def test_stamp_html_file_and_main_round_trip(tmp_path: Path) -> None:
    html_path = tmp_path / "index.html"
    html_path.write_text(
        "<html><head><title>Docs</title></head><body>StackSats Documentation</body></html>",
        encoding="utf-8",
    )

    assert stamp_docs_pages_marker.main(
        [
            "--html-path",
            str(html_path),
            "--commit",
            "feedface",
            "--built-at",
            "2026-03-30T17:00:00Z",
        ]
    ) == 0

    stamped = html_path.read_text(encoding="utf-8")
    assert 'content="feedface"' in stamped
    assert 'content="2026-03-30T17:00:00Z"' in stamped


def test_stamp_html_file_fails_for_missing_or_malformed_html(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.html"
    with pytest.raises(FileNotFoundError):
        stamp_docs_pages_marker.stamp_html_file(
            missing_path,
            commit="abc123",
            built_at="2026-03-30T15:00:00Z",
        )

    malformed_path = tmp_path / "bad.html"
    malformed_path.write_text("<html><body>no head close</body></html>", encoding="utf-8")
    with pytest.raises(ValueError, match="</head>"):
        stamp_docs_pages_marker.stamp_html_file(
            malformed_path,
            commit="abc123",
            built_at="2026-03-30T15:00:00Z",
        )


def test_fetch_html_handles_http_error(monkeypatch) -> None:
    class _HTTPError(urllib.error.HTTPError):
        def __init__(self) -> None:
            super().__init__(
                url="https://example.com",
                code=503,
                msg="service unavailable",
                hdrs=None,
                fp=None,
            )

        def read(self) -> bytes:
            return b"temporarily unavailable"

        def geturl(self) -> str:
            return "https://example.com/final"

    def _raise_http_error(request, timeout):  # noqa: ARG001
        raise _HTTPError()

    monkeypatch.setattr(check_docs_pages_canary.urllib.request, "urlopen", _raise_http_error)

    status_code, final_url, body = check_docs_pages_canary._fetch_html(
        url="https://example.com",
        timeout_seconds=5,
    )

    assert status_code == 503
    assert final_url == "https://example.com/final"
    assert body == "temporarily unavailable"


def test_check_once_succeeds_when_commit_marker_matches(monkeypatch) -> None:
    def _fake_fetch_html(*, url: str, timeout_seconds: int) -> tuple[int, str, str]:
        assert "_stacksats_canary=" in url
        assert timeout_seconds == 7
        return (
            200,
            "https://hypertrial.github.io/stacksats/",
            (
                "<html><head>"
                '<meta name="stacksats-docs-commit" content="abc123">'
                "</head><body>StackSats Documentation</body></html>"
            ),
        )

    monkeypatch.setattr(check_docs_pages_canary, "_fetch_html", _fake_fetch_html)
    monkeypatch.setattr(check_docs_pages_canary.time, "time", lambda: 1234567890)

    check_docs_pages_canary._check_once(
        base_url="https://hypertrial.github.io/stacksats/",
        expected_commit="abc123",
        request_timeout=7,
        attempt=1,
    )


@pytest.mark.parametrize(
    ("status_code", "body", "expected_error"),
    [
        (503, "temporarily unavailable", "Expected HTTP 200"),
        (200, "<html><body>wrong body</body></html>", "missing homepage marker"),
        (
            200,
            "<html><body>StackSats Documentation</body></html>",
            "missing the 'stacksats-docs-commit' meta tag",
        ),
        (
            200,
            (
                "<html><head>"
                '<meta name="stacksats-docs-commit" content="stale">'
                "</head><body>StackSats Documentation</body></html>"
            ),
            "expected freshsha, got stale",
        ),
    ],
)
def test_check_once_fails_with_compact_diagnostics(
    monkeypatch,
    status_code: int,
    body: str,
    expected_error: str,
) -> None:
    monkeypatch.setattr(
        check_docs_pages_canary,
        "_fetch_html",
        lambda *, url, timeout_seconds: (status_code, url, body),  # noqa: ARG005
    )
    monkeypatch.setattr(check_docs_pages_canary.time, "time", lambda: 1234567890)

    with pytest.raises(check_docs_pages_canary.CanaryCheckError, match=expected_error):
        check_docs_pages_canary._check_once(
            base_url="https://hypertrial.github.io/stacksats/",
            expected_commit="freshsha",
            request_timeout=5,
            attempt=2,
        )


def test_run_canary_retries_until_success(monkeypatch) -> None:
    attempts: list[int] = []
    monotonic_values = iter([0.0, 1.0])

    def _fake_check_once(**kwargs):
        attempts.append(int(kwargs["attempt"]))
        if len(attempts) == 1:
            raise check_docs_pages_canary.CanaryCheckError("stale site")

    monkeypatch.setattr(check_docs_pages_canary, "_check_once", _fake_check_once)
    monkeypatch.setattr(check_docs_pages_canary.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(check_docs_pages_canary.time, "sleep", lambda seconds: None)  # noqa: ARG005

    check_docs_pages_canary.run_canary(
        base_url="https://hypertrial.github.io/stacksats/",
        expected_commit="abc123",
        timeout_seconds=30,
        poll_interval_seconds=5,
    )

    assert attempts == [1, 2]


def test_run_canary_times_out_and_main_validates_inputs(monkeypatch) -> None:
    monotonic_values = iter([0.0, 5.0])
    monkeypatch.setattr(
        check_docs_pages_canary,
        "_check_once",
        lambda **kwargs: (_ for _ in ()).throw(  # noqa: ARG005
            check_docs_pages_canary.CanaryCheckError("still stale")
        ),
    )
    monkeypatch.setattr(check_docs_pages_canary.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(check_docs_pages_canary.time, "sleep", lambda seconds: None)  # noqa: ARG005

    with pytest.raises(check_docs_pages_canary.CanaryCheckError, match="still stale"):
        check_docs_pages_canary.run_canary(
            base_url="https://hypertrial.github.io/stacksats/",
            expected_commit="abc123",
            timeout_seconds=3,
            poll_interval_seconds=1,
        )

    with pytest.raises(ValueError, match="timeout_seconds"):
        check_docs_pages_canary.run_canary(
            base_url="https://hypertrial.github.io/stacksats/",
            expected_commit="abc123",
            timeout_seconds=0,
            poll_interval_seconds=1,
        )
    with pytest.raises(ValueError, match="poll_interval_seconds"):
        check_docs_pages_canary.run_canary(
            base_url="https://hypertrial.github.io/stacksats/",
            expected_commit="abc123",
            timeout_seconds=10,
            poll_interval_seconds=0,
        )
    with pytest.raises(ValueError, match="expected_commit"):
        check_docs_pages_canary.run_canary(
            base_url="https://hypertrial.github.io/stacksats/",
            expected_commit="",
            timeout_seconds=10,
            poll_interval_seconds=1,
        )

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        check_docs_pages_canary,
        "run_canary",
        lambda **kwargs: captured.update(kwargs),
    )
    assert check_docs_pages_canary.main(
        [
            "--expected-commit",
            "feedface",
            "--timeout-seconds",
            "11",
            "--poll-interval-seconds",
            "2",
        ]
    ) == 0
    assert captured == {
        "base_url": "https://hypertrial.github.io/stacksats/",
        "expected_commit": "feedface",
        "timeout_seconds": 11,
        "poll_interval_seconds": 2,
    }


def test_docs_build_smoke_stamps_commit_marker(tmp_path: Path) -> None:
    repo_root = _repo_root()
    site_dir = tmp_path / "site"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
            "--site-dir",
            str(site_dir),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    assert stamp_docs_pages_marker.main(
        [
            "--html-path",
            str(site_dir / "index.html"),
            "--commit",
            "smoketestsha",
            "--built-at",
            "2026-03-30T18:00:00Z",
        ]
    ) == 0

    index_html = (site_dir / "index.html").read_text(encoding="utf-8")
    assert 'name="stacksats-docs-commit" content="smoketestsha"' in index_html
    assert "StackSats Documentation" in index_html
