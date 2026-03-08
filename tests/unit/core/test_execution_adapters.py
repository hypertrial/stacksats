from __future__ import annotations

from pathlib import Path

from stacksats.api import DailyOrderReceipt, DailyOrderRequest
from stacksats.execution_adapters import (
    PaperExecutionAdapter,
    load_execution_adapter,
)


class _InlineAdapter:
    def submit_order(self, request: DailyOrderRequest, *, idempotency_key: str) -> DailyOrderReceipt:
        return DailyOrderReceipt(
            status="filled",
            external_order_id=f"inline-{idempotency_key}",
            filled_notional_usd=request.notional_usd,
            filled_quantity_btc=request.quantity_btc,
            fill_price_usd=request.price_usd,
            metadata={"adapter": "inline"},
        )


def _request() -> DailyOrderRequest:
    return DailyOrderRequest(
        strategy_id="test",
        strategy_version="1.0.0",
        run_date="2025-01-01",
        mode="paper",
        weight_today=0.01,
        notional_usd=10.0,
        price_usd=50000.0,
        quantity_btc=0.0002,
        btc_price_col="price_usd",
    )


def test_paper_execution_adapter_is_deterministic() -> None:
    adapter = PaperExecutionAdapter()
    receipt = adapter.submit_order(_request(), idempotency_key="abc123")
    assert receipt.status == "filled"
    assert receipt.external_order_id == "paper-abc123"
    assert receipt.filled_notional_usd == 10.0
    assert receipt.filled_quantity_btc == 0.0002


def test_load_execution_adapter_from_module_path() -> None:
    adapter = load_execution_adapter("tests.unit.core.test_execution_adapters:_InlineAdapter")
    receipt = adapter.submit_order(_request(), idempotency_key="xyz")
    assert receipt.external_order_id == "inline-xyz"


def test_load_execution_adapter_from_file_path(tmp_path: Path) -> None:
    adapter_path = tmp_path / "adapter_impl.py"
    adapter_path.write_text(
        "\n".join(
            [
                "from stacksats.api import DailyOrderReceipt",
                "",
                "class FileAdapter:",
                "    def submit_order(self, request, *, idempotency_key):",
                "        return DailyOrderReceipt(",
                "            status='filled',",
                "            external_order_id='file-' + idempotency_key,",
                "            filled_notional_usd=request.notional_usd,",
                "            filled_quantity_btc=request.quantity_btc,",
                "            fill_price_usd=request.price_usd,",
                "            metadata={'adapter': 'file'}",
                "        )",
            ]
        ),
        encoding="utf-8",
    )
    spec = f"{adapter_path}:FileAdapter"
    adapter = load_execution_adapter(spec)
    receipt = adapter.submit_order(_request(), idempotency_key="filekey")
    assert receipt.external_order_id == "file-filekey"
