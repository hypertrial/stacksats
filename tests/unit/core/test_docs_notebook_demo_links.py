from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urljoin


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _mkdocs_site_url(root: Path) -> str:
    mkdocs_yml = (root / "mkdocs.yml").read_text(encoding="utf-8")
    match = re.search(r"^site_url:\s*(\S+)\s*$", mkdocs_yml, flags=re.MULTILINE)
    assert match, "mkdocs.yml must define site_url"
    return match.group(1).strip()


def _notebook_demo_markdown(root: Path) -> str:
    return (root / "docs" / "start" / "notebook-demo.md").read_text(encoding="utf-8")


def _iframe_src(markdown: str) -> str:
    match = re.search(r"<iframe[\s\S]*?src=\"([^\"]+)\"", markdown)
    assert match, "Notebook demo page must contain iframe src"
    return match.group(1)


def test_notebook_demo_iframe_src_resolves_to_site_assets() -> None:
    root = _repo_root()
    site_url = _mkdocs_site_url(root)
    markdown = _notebook_demo_markdown(root)
    iframe_src = _iframe_src(markdown)

    page_url = urljoin(site_url, "start/notebook-demo/")
    resolved_iframe_url = urljoin(page_url, iframe_src)
    expected_asset_url = urljoin(site_url, "assets/notebooks/model_example_notebook.html")

    assert resolved_iframe_url == expected_asset_url
    assert "/start/assets/" not in resolved_iframe_url


def test_notebook_demo_direct_link_points_to_hosted_notebook_asset() -> None:
    markdown = _notebook_demo_markdown(_repo_root())
    match = re.search(
        r"\[Open the exported notebook directly\]\(([^)]+)\)",
        markdown,
    )
    assert match, "Notebook demo page must include direct notebook link"
    assert (
        match.group(1)
        == "https://hypertrial.github.io/stacksats/assets/notebooks/model_example_notebook.html"
    )
