#!/usr/bin/env python3
"""Lightweight UX structure checks for key docs pages.

This complements link/build checks by enforcing a task-first shape on core pages.
"""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_HEADINGS: dict[str, list[str]] = {
    "docs/index.md": [
        "## Choose Your Path",
        "## Most Common Tasks",
        "## Feedback",
    ],
    "docs/tasks.md": [
        "# I Want To...",
        "## Feedback",
    ],
    "docs/commands.md": [
        "## Most Common Commands (copy/paste)",
        "## 2) Validate Strategy via Strategy Lifecycle CLI",
        "## 3) Run Full Backtest via Strategy Lifecycle CLI",
        "## 4) Export Strategy Artifacts",
        "## Troubleshooting",
        "## Feedback",
    ],
    "docs/start/quickstart.md": [
        "## Success Criteria",
        "## Troubleshooting",
        "## Feedback",
    ],
    "docs/start/first-strategy-run.md": [
        "## Success Criteria",
        "## 5) Troubleshooting",
        "## Next Steps",
        "## Feedback",
    ],
    "docs/start/minimal-strategy-examples.md": [
        "# Minimal Strategy Examples",
        "## Example A: `propose_weight(state)` style",
        "## Example B: `build_target_profile(...)` style",
        "## Success Criteria",
        "## Feedback",
    ],
    "docs/migration.md": [
        "## Old -> New Mapping",
        "## Upgrade Checklist",
        "## Feedback",
    ],
    "docs/faq.md": [
        "# FAQ",
        "## Strategy authoring",
        "## CLI and outputs",
        "## Migration and compatibility",
        "## Docs feedback workflow",
        "## Feedback",
    ],
}

REQUIRED_SUBSTRINGS: dict[str, list[str]] = {
    "docs/migration.md": [
        "compute_weights_shared",
        "compute_weights_with_features",
        "BACKTEST_END",
        "get_backtest_end()",
        "generate_date_ranges(start, end, min_length_days)",
        "generate_date_ranges(start, end)",
    ],
    "docs/commands.md": [
        "--start-date",
        "--end-date",
        "output/<strategy_id>/<version>/<run_id>/",
    ],
    "docs/start/minimal-strategy-examples.md": [
        "propose_weight",
        "build_target_profile",
    ],
    "docs/faq.md": [
        "template=docs_feedback.md",
        "label%3Adocumentation",
    ],
}

FEEDBACK_LINK_PAGES = [
    "docs/index.md",
    "docs/tasks.md",
    "docs/commands.md",
    "docs/start/quickstart.md",
    "docs/start/first-strategy-run.md",
    "docs/start/minimal-strategy-examples.md",
    "docs/migration.md",
    "docs/faq.md",
    "docs/model_backtest.md",
    "docs/reference/strategy-timeseries.md",
]

FEEDBACK_TOKEN = "template=docs_feedback.md"


def load(rel: str) -> str:
    path = ROOT / rel
    if not path.exists():
        raise FileNotFoundError(f"missing file: {rel}")
    return path.read_text(encoding="utf-8")


def main() -> int:
    errors: list[str] = []

    for rel, headings in REQUIRED_HEADINGS.items():
        try:
            text = load(rel)
        except FileNotFoundError as exc:
            errors.append(str(exc))
            continue
        for heading in headings:
            if heading not in text:
                errors.append(f"{rel}: missing heading '{heading}'")

    for rel, required in REQUIRED_SUBSTRINGS.items():
        text = load(rel)
        for token in required:
            if token not in text:
                errors.append(f"{rel}: missing required content '{token}'")

    tasks_text = load("docs/tasks.md")
    task_heading_count = len(re.findall(r"^## I want to ", tasks_text, flags=re.MULTILINE))
    if task_heading_count < 5:
        errors.append(
            "docs/tasks.md: expected at least 5 'I want to ...' task sections"
        )

    for subsection in [
        "### Prerequisites",
        "### Command",
        "### Expected output",
        "### Troubleshooting",
        "### Next step",
    ]:
        if tasks_text.count(subsection) < 4:
            errors.append(
                f"docs/tasks.md: expected repeated subsection '{subsection}' in task blocks"
            )

    for rel in FEEDBACK_LINK_PAGES:
        text = load(rel)
        if FEEDBACK_TOKEN not in text:
            errors.append(f"{rel}: missing docs feedback link token '{FEEDBACK_TOKEN}'")

    if errors:
        print("Docs UX check failed:")
        for err in errors:
            print(f" - {err}")
        return 1

    print("Docs UX check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
