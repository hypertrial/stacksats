from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _notebook_demo_markdown(root: Path) -> str:
    return (root / "docs" / "start" / "notebook-demo.md").read_text(encoding="utf-8")


def test_notebook_demo_does_not_embed_iframe() -> None:
    markdown = _notebook_demo_markdown(_repo_root())
    assert "<iframe" not in markdown


def test_notebook_demo_does_not_link_removed_hosted_notebook_asset() -> None:
    markdown = _notebook_demo_markdown(_repo_root())
    assert "model_example_notebook.html" not in markdown


def test_notebook_demo_points_to_maintained_workflow_docs() -> None:
    markdown = _notebook_demo_markdown(_repo_root())
    assert "[Quickstart](quickstart.md)" in markdown
    assert "[CLI Commands](../commands.md)" in markdown
