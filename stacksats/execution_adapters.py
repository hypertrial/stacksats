"""Execution adapter interface and built-in paper adapter."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

from .api import DailyOrderReceipt, DailyOrderRequest


@runtime_checkable
class ExecutionAdapter(Protocol):
    """Interface contract for live/paper execution adapters."""

    def submit_order(
        self,
        request: DailyOrderRequest,
        *,
        idempotency_key: str,
    ) -> DailyOrderReceipt:
        """Submit today's order request and return execution receipt."""


class PaperExecutionAdapter:
    """Deterministic in-process paper execution adapter."""

    def submit_order(
        self,
        request: DailyOrderRequest,
        *,
        idempotency_key: str,
    ) -> DailyOrderReceipt:
        return DailyOrderReceipt(
            status="filled",
            external_order_id=f"paper-{idempotency_key}",
            filled_notional_usd=float(request.notional_usd),
            filled_quantity_btc=float(request.quantity_btc),
            fill_price_usd=float(request.price_usd),
            metadata={"adapter": "paper", "simulated": "true"},
        )


def _load_module(module_or_path: str):
    if module_or_path.endswith(".py"):
        file_path = Path(module_or_path).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Adapter file not found: {file_path}")
        module_hash = hashlib.sha1(str(file_path).encode("utf-8")).hexdigest()[:10]
        module_name = f"stacksats_execution_adapter_{module_hash}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load adapter module spec from file: {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module

    return importlib.import_module(module_or_path)


def load_execution_adapter(spec: str) -> ExecutionAdapter:
    """Load an execution adapter from `module_or_path:ClassName`."""
    module_or_path, sep, class_name = spec.rpartition(":")
    if not sep or not module_or_path or not class_name:
        raise ValueError("Invalid adapter spec. Use format 'module_or_path:ClassName'.")
    module = _load_module(module_or_path)
    try:
        adapter_cls = getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Class '{class_name}' not found in '{module_or_path}'."
        ) from exc

    adapter = adapter_cls()
    submit_order = getattr(adapter, "submit_order", None)
    if not callable(submit_order):
        raise TypeError("Execution adapter must define callable submit_order(...).")
    return adapter


__all__ = ["ExecutionAdapter", "PaperExecutionAdapter", "load_execution_adapter"]
