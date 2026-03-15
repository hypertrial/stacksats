"""AST-based linting for causal strategy contract enforcement."""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StrategyLintFinding:
    severity: str
    code: str
    message: str
    lineno: int | None = None


def lint_strategy_class(strategy_cls: type) -> list[StrategyLintFinding]:
    """Return lint findings for a strategy class."""
    try:
        source = inspect.getsource(strategy_cls)
    except (OSError, TypeError):
        return [
            StrategyLintFinding(
                severity="warning",
                code="source-unavailable",
                message=(
                    "Strategy source could not be inspected for causal lint checks."
                ),
                lineno=None,
            )
        ]
    tree = ast.parse(textwrap.dedent(source))
    visitor = _StrategyLintVisitor()
    visitor.visit(tree)
    return visitor.findings


def summarize_lint_findings(findings: list[StrategyLintFinding]) -> tuple[list[str], list[str]]:
    """Split lint findings into hard errors and warnings."""
    errors = []
    warnings = []
    for finding in findings:
        line = f"[{finding.code}] {finding.message}"
        if finding.lineno is not None:
            line = f"line {finding.lineno}: {line}"
        if finding.severity == "error":
            errors.append(line)
        else:
            warnings.append(line)
    return errors, warnings


class _StrategyLintVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.findings: list[StrategyLintFinding] = []

    def visit_Call(self, node: ast.Call) -> None:
        attr_name = node.func.attr if isinstance(node.func, ast.Attribute) else None
        dotted_name = _attribute_name(node.func)

        if attr_name == "shift":
            if _is_negative_integer(_call_argument(node, 0, "periods")):
                self._error(
                    node,
                    "negative-shift",
                    "Negative shift detected in strategy code.",
                )

        if attr_name == "rolling" or (attr_name and "rolling" in attr_name):
            center_arg = _call_keyword(node, "center")
            if _is_true_literal(center_arg):
                self._error(
                    node,
                    "centered-rolling",
                    "Centered rolling windows are not allowed in strategy code.",
                )

        if dotted_name in {
            "open",
            "pd.read_csv",
            "pd.read_parquet",
            "pl.read_csv",
            "pl.read_parquet",
            "polars.read_csv",
            "polars.read_parquet",
            "sqlite3.connect",
            "requests.get",
            "requests.post",
            "requests.put",
            "requests.delete",
            "requests.request",
            "httpx.get",
            "httpx.post",
            "httpx.request",
            "urllib.request.urlopen",
        }:
            self._error(
                node,
                "external-io",
                f"External I/O is not allowed in strategy code: {dotted_name}.",
            )

        if dotted_name.endswith(".read_text") or dotted_name.endswith(".read_bytes"):
            self._error(
                node,
                "path-io",
                f"Path-based file I/O is not allowed in strategy code: {dotted_name}.",
            )

        if attr_name == "tail":
            self._warning(
                node,
                "tail-usage",
                "tail(...) can hide forward-looking assumptions; review manually.",
            )
        if attr_name in {"rank", "quantile"} and not _is_rolling_aggregation(node):
            self._warning(
                node,
                "global-ranking",
                f"{attr_name}(...) outside a rolling window may leak full-sample information.",
            )

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Attribute) and node.value.attr == "iloc":
            if _is_negative_integer(node.slice):
                self._warning(
                    node,
                    "iloc-negative",
                    "iloc[-1] style indexing can hide end-of-sample leakage.",
                )
        self.generic_visit(node)

    def _error(self, node: ast.AST, code: str, message: str) -> None:
        self.findings.append(
            StrategyLintFinding(
                severity="error",
                code=code,
                message=message,
                lineno=getattr(node, "lineno", None),
            )
        )

    def _warning(self, node: ast.AST, code: str, message: str) -> None:
        self.findings.append(
            StrategyLintFinding(
                severity="warning",
                code=code,
                message=message,
                lineno=getattr(node, "lineno", None),
            )
        )


def _attribute_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _call_keyword(node: ast.Call, name: str) -> ast.AST | None:
    for keyword in node.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _call_argument(node: ast.Call, position: int, keyword_name: str) -> ast.AST | None:
    if len(node.args) > position:
        return node.args[position]
    return _call_keyword(node, keyword_name)


def _is_true_literal(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _is_negative_integer(node: ast.AST | None) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value < 0
    return False


def _is_rolling_aggregation(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute):
        return False
    value = node.func.value
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "rolling"
    )
