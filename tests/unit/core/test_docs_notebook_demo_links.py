from __future__ import annotations

import re
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _notebook_demo_markdown(root: Path) -> str:
    return (root / "docs" / "start" / "notebook-demo.md").read_text(encoding="utf-8")


def test_notebook_demo_does_not_embed_iframe() -> None:
    markdown = _notebook_demo_markdown(_repo_root())
    assert "<iframe" not in markdown


def test_notebook_demo_direct_link_points_to_hosted_notebook_asset() -> None:
    markdown = _notebook_demo_markdown(_repo_root())
    match = re.search(
        r"\[Open browser-safe exported notebook\]\(([^)]+)\)",
        markdown,
    )
    assert match, "Notebook demo page must include browser-safe notebook link"
    assert (
        match.group(1)
        == "https://hypertrial.github.io/stacksats/assets/notebooks/model_example_notebook_browser.html"
    )
