from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_agent_api_doc_page_contains_curl_examples_and_discovery_route() -> None:
    markdown = (_repo_root() / "docs" / "run" / "agent-api.md").read_text(encoding="utf-8")

    assert "stacksats serve agent-api" in markdown
    assert "curl -sS http://127.0.0.1:8000/.well-known/agent-integration.json" in markdown
    assert "POST /v1/decisions/daily" in markdown
    assert "POST /v1/executions/receipts" in markdown
    assert "X-Request-ID" in markdown
    assert "Rotate bearer tokens" in markdown
    assert "restart or roll the service" in markdown


def test_commands_and_public_api_docs_reference_agent_api_service() -> None:
    commands = (_repo_root() / "docs" / "commands.md").read_text(encoding="utf-8")
    public_api = (_repo_root() / "docs" / "reference" / "public-api.md").read_text(
        encoding="utf-8"
    )

    assert "run/agent-api.md" in commands
    assert "stacksats serve agent-api" in commands
    assert "documented hosted HTTP service" in public_api
    assert "POST /v1/decisions/daily" in public_api
