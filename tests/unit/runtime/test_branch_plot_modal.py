from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pandas as pd

import stacksats.modal_app as modal_app
from stacksats.plot_mvrv import main as plot_mvrv_main
from stacksats.plot_weights import main as plot_weights_main, plot_dca_weights


def test_plot_mvrv_main_returns_nonzero_on_value_error(
    monkeypatch,
) -> None:
    bad_df = pd.DataFrame(
        {"PriceUSD_coinmetrics": [100.0, 101.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    monkeypatch.setattr("stacksats.plot_mvrv.fetch_coinmetrics_btc_csv", lambda: bad_df)
    monkeypatch.setattr(sys, "argv", ["plot_mvrv.py"])

    assert plot_mvrv_main() == 1


def test_plot_mvrv_main_returns_nonzero_on_unexpected_error(
    monkeypatch,
) -> None:
    def _boom():
        raise RuntimeError("unexpected")

    monkeypatch.setattr("stacksats.plot_mvrv.fetch_coinmetrics_btc_csv", _boom)
    monkeypatch.setattr(sys, "argv", ["plot_mvrv.py"])

    assert plot_mvrv_main() == 1


def test_plot_weights_handles_future_only_data_and_normalizes(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    df = pd.DataFrame(
        {
            "DCA_date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "weight": [2.0, 2.0, 2.0],
            "btc_usd": [None, None, None],
            "id": [1, 2, 3],
        }
    )
    monkeypatch.setattr("stacksats.plot_weights.plt.savefig", lambda *_args, **_kwargs: None)

    plot_dca_weights(
        df,
        start_date="2024-01-01",
        end_date="2024-01-03",
        output_path=str(tmp_path / "weights.svg"),
    )

    output = capsys.readouterr().out
    assert "Future weights (no price): 3" in output
    assert "Weights normalized from 6.000000 to 1.0" in output


def test_plot_weights_main_uses_oldest_range_when_no_args(
    monkeypatch,
) -> None:
    conn = MagicMock()
    captured: dict[str, str] = {}

    def _fake_plot(df, start_date, end_date, output_path):
        del df, output_path
        captured["start_date"] = start_date
        captured["end_date"] = end_date

    monkeypatch.setattr("stacksats.plot_weights.get_db_connection", lambda: conn)
    monkeypatch.setattr(
        "stacksats.plot_weights.get_oldest_date_range",
        lambda _conn: ("2024-01-01", "2024-12-31"),
    )
    monkeypatch.setattr(
        "stacksats.plot_weights.fetch_weights_for_date_range",
        lambda _conn, _start, _end: pd.DataFrame(
            {
                "DCA_date": pd.date_range("2024-01-01", periods=2, freq="D"),
                "weight": [0.5, 0.5],
                "btc_usd": [50000.0, None],
                "id": [1, 2],
            }
        ),
    )
    monkeypatch.setattr("stacksats.plot_weights.plot_dca_weights", _fake_plot)
    monkeypatch.setattr(sys, "argv", ["plot_weights.py"])

    plot_weights_main()

    assert captured["start_date"] == "2024-01-01"
    assert captured["end_date"] == "2024-12-31"
    assert conn.close.called


def test_modal_local_entrypoint_prints_export_summary(monkeypatch, capsys) -> None:
    final_df = pd.DataFrame({"id": [1, 2], "weight": [0.4, 0.6]})
    metadata = {
        "rows": 2,
        "date_ranges": 1,
        "range_start": "2024-01-01",
        "range_end": "2024-12-31",
        "export_date": "2024-01-15",
    }
    monkeypatch.setattr(modal_app.run_export, "remote", lambda: (final_df, metadata))

    modal_app.main()

    output = capsys.readouterr().out
    assert "Running export via Modal..." in output
    assert "Successfully exported 2 rows" in output
