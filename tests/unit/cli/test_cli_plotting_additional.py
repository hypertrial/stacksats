from __future__ import annotations

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from stacksats import cli
from stacksats.matplotlib_setup import configure_matplotlib_env
from stacksats.plot_mvrv import plot_mvrv_metrics
from stacksats.plot_weights import main as main_weights


def test_cli_validate_exits_nonzero_on_failed_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResult:
        passed = False
        messages = ["failed"]

        @staticmethod
        def summary() -> str:
            return "Validation FAILED"

    class FakeRunner:
        def validate(self, strategy, config):
            del strategy, config
            return FakeResult()

    monkeypatch.setattr(cli, "StrategyRunner", lambda: FakeRunner())
    monkeypatch.setattr(cli, "load_strategy", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stacksats",
            "strategy",
            "validate",
            "--strategy",
            "dummy.py:Dummy",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 1


def test_plot_mvrv_metrics_raises_when_required_column_missing() -> None:
    df = pd.DataFrame({"Other": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3))

    with pytest.raises(ValueError, match="Missing required column: CapMVRVCur"):
        plot_mvrv_metrics(df)


def test_plot_mvrv_metrics_raises_when_filtered_data_is_empty(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "CapMVRVCur": [1.0, 1.1, 1.2],
            "CapMVRVZ": [0.0, 0.1, 0.2],
        },
        index=pd.date_range("2024-01-01", periods=3),
    )

    with pytest.raises(ValueError, match="No data available for the specified date range"):
        plot_mvrv_metrics(
            df,
            start_date="2030-01-01",
            end_date="2030-12-31",
            output_path=str(tmp_path / "out.svg"),
        )


def test_plot_weights_list_option_prints_ranges_and_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_conn = MagicMock()
    monkeypatch.setattr("stacksats.plot_weights.get_db_connection", lambda: mock_conn)
    monkeypatch.setattr(
        "stacksats.plot_weights.get_date_range_options",
        lambda conn: pd.DataFrame(
            {
                "start_date": pd.to_datetime(["2024-01-01"]),
                "end_date": pd.to_datetime(["2024-12-31"]),
                "count": [366],
            }
        ),
    )
    monkeypatch.setattr(sys, "argv", ["stacksats.plot_weights.py", "--list"])

    main_weights()

    assert mock_conn.close.called


def test_plot_weights_invalid_range_exits_with_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_conn = MagicMock()
    monkeypatch.setattr("stacksats.plot_weights.get_db_connection", lambda: mock_conn)
    monkeypatch.setattr("stacksats.plot_weights.validate_date_range", lambda *args: False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["stacksats.plot_weights.py", "2024-01-01", "2024-12-31"],
    )

    with pytest.raises(SystemExit) as excinfo:
        main_weights()
    assert excinfo.value.code == 1
    assert mock_conn.close.called


def test_configure_matplotlib_env_uses_temp_fallback_when_home_paths_unwritable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", "/definitely/not/writable")
    monkeypatch.setenv("MPLCONFIGDIR", "/definitely/not/writable")

    def fake_writable(path: Path) -> bool:
        return "stacksats-cache" in str(path)

    monkeypatch.setattr("stacksats.matplotlib_setup._is_writable_dir", fake_writable)

    configure_matplotlib_env()

    assert "stacksats-cache" in str(Path(os.environ["XDG_CACHE_HOME"]))
    assert "stacksats-cache" in str(Path(os.environ["MPLCONFIGDIR"]))


def test_configure_matplotlib_env_preserves_existing_writable_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    xdg = str(tmp_path / "xdg")
    mpl = str(tmp_path / "mpl")
    monkeypatch.setenv("XDG_CACHE_HOME", xdg)
    monkeypatch.setenv("MPLCONFIGDIR", mpl)
    monkeypatch.setattr("stacksats.matplotlib_setup._is_writable_dir", lambda path: True)

    configure_matplotlib_env()

    assert os.environ["XDG_CACHE_HOME"] == xdg
    assert os.environ["MPLCONFIGDIR"] == mpl
