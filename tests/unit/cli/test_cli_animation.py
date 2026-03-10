from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from stacksats import cli


def test_cli_strategy_animate_success_path(monkeypatch, capsys, tmp_path: Path) -> None:
    backtest_json = tmp_path / "backtest_result.json"
    backtest_json.write_text("{}", encoding="utf-8")

    class FakeBacktestResult:
        def animate(
            self,
            output_dir: str,
            *,
            fps: int,
            width: int,
            height: int,
            max_frames: int,
            filename: str,
            window_mode: str,
            source_backtest_json: str | Path | None,
        ):
            assert Path(output_dir) == tmp_path
            assert fps == 8
            assert width == 640
            assert height == 360
            assert max_frames == 12
            assert filename == "demo.gif"
            assert window_mode == "rolling"
            assert source_backtest_json == str(backtest_json)
            gif_path = Path(output_dir) / filename
            gif_path.write_bytes(b"GIF89a")
            manifest_path = Path(output_dir) / "animation_manifest.json"
            manifest_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
            return {"gif": str(gif_path), "manifest_json": str(manifest_path)}

    monkeypatch.setattr(cli, "_backtest_result_from_json", lambda path: FakeBacktestResult())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stacksats",
            "strategy",
            "animate",
            "--backtest-json",
            str(backtest_json),
            "--fps",
            "8",
            "--width",
            "640",
            "--height",
            "360",
            "--max-frames",
            "12",
            "--output-name",
            "demo.gif",
            "--window-mode",
            "rolling",
        ],
    )

    cli.main()
    out = capsys.readouterr().out
    assert '"gif"' in out
    assert "Saved:" in out


def test_cli_strategy_animate_does_not_load_strategy(monkeypatch, tmp_path: Path) -> None:
    backtest_json = tmp_path / "backtest_result.json"
    backtest_json.write_text("{}", encoding="utf-8")

    class FakeBacktestResult:
        def animate(self, output_dir: str, **kwargs):
            del kwargs
            output = Path(output_dir)
            gif_path = output / "strategy_vs_uniform_hd.gif"
            gif_path.write_bytes(b"GIF89a")
            manifest_path = output / "animation_manifest.json"
            manifest_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
            return {"gif": str(gif_path), "manifest_json": str(manifest_path)}

    monkeypatch.setattr(cli, "_backtest_result_from_json", lambda path: FakeBacktestResult())
    monkeypatch.setattr(
        cli,
        "load_strategy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("animate should not load strategy")
        ),
    )
    old_argv = sys.argv
    sys.argv = [
        "stacksats",
        "strategy",
        "animate",
        "--backtest-json",
        str(backtest_json),
    ]
    try:
        cli.main()
    finally:
        sys.argv = old_argv


def test_cli_strategy_animate_fails_for_malformed_json(capsys, tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not-valid-json", encoding="utf-8")
    argv = [
        "stacksats",
        "strategy",
        "animate",
        "--backtest-json",
        str(bad_json),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(SystemExit) as raised:
            cli.main()
    finally:
        sys.argv = old_argv
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert "Invalid backtest JSON" in err


def test_cli_strategy_animate_fails_for_missing_window_data(capsys, tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text(json.dumps({"summary_metrics": {}}), encoding="utf-8")
    argv = [
        "stacksats",
        "strategy",
        "animate",
        "--backtest-json",
        str(bad_json),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(SystemExit) as raised:
            cli.main()
    finally:
        sys.argv = old_argv
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert "window_level_data" in err
