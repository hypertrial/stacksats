#!/usr/bin/env python3
"""UX structure checks for core docs pages.

This validates intent-oriented structure and catches docs drift:
- key section intent across high-traffic pages
- task page workflow pattern presence
- command index size and snippet duplication drift
"""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

FEEDBACK_TOKEN = "template=docs_feedback.md"

REQUIRED_HEADING_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "docs/index.md": [
        ("choose path routing", r"^##\s+Choose Your Path$"),
        ("quick routes", r"^##\s+Start in 2 Clicks$"),
        ("feedback section", r"^##\s+Feedback$"),
    ],
    "docs/tasks.md": [
        ("task hub heading", r"^#\s+I Want To"),
        ("feedback section", r"^##\s+Feedback$"),
    ],
    "docs/commands.md": [
        ("command index heading", r"^#\s+Command Index$"),
        ("command pages section", r"^##\s+Command Pages$"),
        ("troubleshooting section", r"^##\s+Troubleshooting$"),
        ("feedback section", r"^##\s+Feedback$"),
        (
            "compatibility validate anchor",
            r"^##\s+(?:\d+\)\s+)?Validate Strategy via Strategy Lifecycle CLI$",
        ),
        (
            "compatibility backtest anchor",
            r"^##\s+(?:\d+\)\s+)?Run Full Backtest via Strategy Lifecycle CLI$",
        ),
        (
            "compatibility export anchor",
            r"^##\s+(?:\d+\)\s+)?Export Strategy Artifacts$",
        ),
    ],
    "docs/start/quickstart.md": [
        ("success criteria section", r"^##\s+Success Criteria$"),
        ("troubleshooting section", r"^##\s+Troubleshooting$"),
        ("feedback section", r"^##\s+Feedback$"),
    ],
    "docs/start/first-strategy-run.md": [
        ("success criteria section", r"^##\s+Success Criteria$"),
        ("next steps section", r"^##\s+Next Steps$"),
        ("feedback section", r"^##\s+Feedback$"),
    ],
    "docs/start/minimal-strategy-examples.md": [
        ("page heading", r"^#\s+Minimal Strategy Examples$"),
        ("example A section", r"^##\s+Example A:\s+.*propose_weight"),
        ("example B section", r"^##\s+Example B:\s+.*build_target_profile"),
        ("success criteria section", r"^##\s+Success Criteria$"),
        ("feedback section", r"^##\s+Feedback$"),
    ],
    "docs/migration.md": [
        ("mapping section", r"^##\s+Old -> New Mapping$"),
        ("upgrade checklist", r"^##\s+Upgrade Checklist$"),
        ("feedback section", r"^##\s+Feedback$"),
    ],
    "docs/faq.md": [
        ("page heading", r"^#\s+FAQ$"),
        ("strategy authoring section", r"^##\s+Strategy authoring$"),
        ("cli outputs section", r"^##\s+CLI and outputs$"),
        ("migration section", r"^##\s+Migration and compatibility$"),
        ("feedback workflow section", r"^##\s+Docs feedback workflow$"),
        ("feedback section", r"^##\s+Feedback$"),
    ],
}

COMMAND_PAGE_PATHS = [
    "docs/run/validate.md",
    "docs/run/backtest.md",
    "docs/run/export.md",
    "docs/run/run-daily.md",
    "docs/run/animate.md",
]

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
    *COMMAND_PAGE_PATHS,
]


def load(rel: str) -> str:
    path = ROOT / rel
    if not path.exists():
        raise FileNotFoundError(f"missing file: {rel}")
    return path.read_text(encoding="utf-8")


def line_count(rel: str) -> int:
    return len(load(rel).splitlines())


def _extract_bash_blocks(text: str) -> list[str]:
    blocks = re.findall(r"```bash\s*(.*?)```", text, flags=re.DOTALL)
    normalized: list[str] = []
    for block in blocks:
        lines = [line.rstrip() for line in block.strip().splitlines()]
        lines = [line for line in lines if line and not line.strip().startswith("#")]
        if len(lines) < 2:
            continue
        normalized.append("\n".join(lines))
    return normalized


def main() -> int:
    errors: list[str] = []

    for rel, required_patterns in REQUIRED_HEADING_PATTERNS.items():
        try:
            text = load(rel)
        except FileNotFoundError as exc:
            errors.append(str(exc))
            continue
        for label, pattern in required_patterns:
            if not re.search(pattern, text, flags=re.MULTILINE):
                errors.append(f"{rel}: missing required section intent '{label}'")

    first_strategy_text = load("docs/start/first-strategy-run.md")
    if not re.search(
        r"^##\s+(?:\d+\)\s+)?Troubleshooting$",
        first_strategy_text,
        re.MULTILINE,
    ):
        errors.append("docs/start/first-strategy-run.md: missing troubleshooting section")

    tasks_text = load("docs/tasks.md")
    task_heading_count = len(
        re.findall(r"^##\s+I want to ", tasks_text, flags=re.MULTILINE | re.IGNORECASE)
    )
    if task_heading_count < 7:
        errors.append("docs/tasks.md: expected at least 7 'I want to ...' sections")

    command_subsections = len(
        re.findall(r"^###\s+Command[s]?$", tasks_text, flags=re.MULTILINE | re.IGNORECASE)
    )
    if command_subsections < 7:
        errors.append(
            "docs/tasks.md: expected repeated command sections across task blocks"
        )
    if (
        len(
            re.findall(
                r"^###\s+Troubleshooting$", tasks_text, flags=re.MULTILINE | re.IGNORECASE
            )
        )
        < 7
    ):
        errors.append(
            "docs/tasks.md: expected repeated troubleshooting sections across task blocks"
        )
    if (
        len(
            re.findall(
                r"^###\s+Next step[s]?$", tasks_text, flags=re.MULTILINE | re.IGNORECASE
            )
        )
        < 7
    ):
        errors.append("docs/tasks.md: expected repeated next-step sections across task blocks")

    commands_text = load("docs/commands.md")
    for token in [
        "run/validate.md",
        "run/backtest.md",
        "run/export.md",
        "run/run-daily.md",
        "run/animate.md",
    ]:
        if token not in commands_text:
            errors.append(f"docs/commands.md: missing command-page link '{token}'")

    if line_count("docs/commands.md") > 220:
        errors.append("docs/commands.md: command index should stay concise (<= 220 lines)")

    for rel in COMMAND_PAGE_PATHS:
        if line_count(rel) > 220:
            errors.append(f"{rel}: command page should stay concise (<= 220 lines)")

    command_blocks = set(_extract_bash_blocks(commands_text))
    tasks_blocks = set(_extract_bash_blocks(tasks_text))
    duplicates = sorted(command_blocks.intersection(tasks_blocks))
    if duplicates:
        errors.append(
            "docs/commands.md and docs/tasks.md contain duplicated long bash snippets; "
            "keep commands canonical in docs/run/*.md and task snippets minimal."
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
