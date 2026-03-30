#!/usr/bin/env python3
"""External canary check for deployed StackSats GitHub Pages docs."""

from __future__ import annotations

import argparse
import time
from typing import Final
import urllib.error
import urllib.parse
import urllib.request
import re

KNOWN_CONTENT_MARKER: Final[str] = "StackSats Documentation"
COMMIT_META_NAME: Final[str] = "stacksats-docs-commit"
COMMIT_META_RE = re.compile(
    rf'<meta\s+name="{re.escape(COMMIT_META_NAME)}"\s+content="([^"]+)">',
    flags=re.IGNORECASE,
)


class CanaryCheckError(RuntimeError):
    """Raised when the external docs canary fails."""


def _compact_snippet(text: str, *, limit: int = 180) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _cache_busted_url(base_url: str, *, attempt: int) -> str:
    parts = urllib.parse.urlsplit(base_url)
    query_pairs = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query_pairs.append(("_stacksats_canary", f"{int(time.time())}-{attempt}"))
    return urllib.parse.urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path or "/",
            urllib.parse.urlencode(query_pairs),
            parts.fragment,
        )
    )


def _fetch_html(*, url: str, timeout_seconds: int) -> tuple[int, str, str]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/html",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
            return int(response.status), str(response.geturl()), body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return int(exc.code), str(exc.geturl()), body
    except urllib.error.URLError as exc:
        raise CanaryCheckError(f"Request failed for {url}: {exc}") from exc


def _extract_commit_marker(html: str) -> str | None:
    match = COMMIT_META_RE.search(html)
    if match is None:
        return None
    return match.group(1)


def _check_once(*, base_url: str, expected_commit: str, request_timeout: int, attempt: int) -> None:
    status_code, final_url, body = _fetch_html(
        url=_cache_busted_url(base_url, attempt=attempt),
        timeout_seconds=request_timeout,
    )
    if status_code != 200:
        raise CanaryCheckError(
            f"Expected HTTP 200 from {final_url}, got {status_code}. "
            f"Body: {_compact_snippet(body)}"
        )
    if KNOWN_CONTENT_MARKER not in body:
        raise CanaryCheckError(
            f"Live docs response from {final_url} is missing homepage marker "
            f"{KNOWN_CONTENT_MARKER!r}. Body: {_compact_snippet(body)}"
        )
    live_commit = _extract_commit_marker(body)
    if live_commit is None:
        raise CanaryCheckError(
            f"Live docs response from {final_url} is missing the {COMMIT_META_NAME!r} meta tag. "
            f"Body: {_compact_snippet(body)}"
        )
    if live_commit != expected_commit:
        raise CanaryCheckError(
            f"Live docs commit marker mismatch at {final_url}: "
            f"expected {expected_commit}, got {live_commit}."
        )


def run_canary(
    *,
    base_url: str,
    expected_commit: str,
    timeout_seconds: int,
    poll_interval_seconds: int,
) -> None:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be positive.")
    if not expected_commit.strip():
        raise ValueError("expected_commit must be non-empty.")

    deadline = time.monotonic() + timeout_seconds
    attempt = 0
    last_error = "Pages canary did not run."
    request_timeout = min(15, timeout_seconds)

    while True:
        attempt += 1
        try:
            _check_once(
                base_url=base_url,
                expected_commit=expected_commit,
                request_timeout=request_timeout,
                attempt=attempt,
            )
            return
        except CanaryCheckError as exc:
            last_error = str(exc)
        if time.monotonic() >= deadline:
            raise CanaryCheckError(
                f"Pages canary failed after {attempt} attempt(s) over {timeout_seconds}s. "
                f"{last_error}"
            ) from None
        time.sleep(poll_interval_seconds)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the live StackSats GitHub Pages site is available and fresh.",
    )
    parser.add_argument(
        "--base-url",
        default="https://hypertrial.github.io/stacksats/",
        help="Public docs base URL to fetch.",
    )
    parser.add_argument(
        "--expected-commit",
        required=True,
        help="Expected deployed commit SHA marker.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="Maximum total wait time before the canary fails.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=5,
        help="Delay between retries when the live site is stale or unavailable.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_canary(
        base_url=str(args.base_url),
        expected_commit=str(args.expected_commit),
        timeout_seconds=int(args.timeout_seconds),
        poll_interval_seconds=int(args.poll_interval_seconds),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
