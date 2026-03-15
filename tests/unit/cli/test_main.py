"""Tests for backtest.py - Metrics export."""

import datetime as dt
import json
import os

import polars as pl

from stacksats.backtest import export_metrics_json
from stacksats.prelude import parse_window_dates


# -----------------------------------------------------------------------------
# parse_window_dates() Tests
# -----------------------------------------------------------------------------


class TestParseWindowDates:
    """Tests for the parse_window_dates function."""

    def test_parse_standard_format(self):
        """Test parsing standard window label format."""
        label = "2024-01-01 → 2024-12-31"
        result = parse_window_dates(label)

        assert isinstance(result, dt.datetime)
        assert result == dt.datetime(2024, 1, 1)

    def test_parse_extracts_start_date(self):
        """Test that only start date is extracted."""
        label = "2023-06-15 → 2024-06-15"
        result = parse_window_dates(label)

        assert result == dt.datetime(2023, 6, 15)

    def test_parse_different_years(self):
        """Test parsing label spanning different years."""
        label = "2020-01-01 → 2021-01-01"
        result = parse_window_dates(label)

        assert result.year == 2020
        assert result.month == 1
        assert result.day == 1


# -----------------------------------------------------------------------------
# export_metrics_json() Tests
# -----------------------------------------------------------------------------


class TestExportMetricsJson:
    """Tests for the export_metrics_json function."""

    def test_creates_file(self, sample_spd_df, sample_metrics, temp_output_dir):
        """Test that JSON file is created."""
        export_metrics_json(sample_spd_df, sample_metrics, temp_output_dir)

        output_path = os.path.join(temp_output_dir, "metrics.json")
        assert os.path.exists(output_path)

    def test_valid_json(self, sample_spd_df, sample_metrics, temp_output_dir):
        """Test that output is valid JSON."""
        export_metrics_json(sample_spd_df, sample_metrics, temp_output_dir)

        output_path = os.path.join(temp_output_dir, "metrics.json")
        with open(output_path, "r") as f:
            data = json.load(f)

        assert isinstance(data, dict)

    def test_has_required_keys(self, sample_spd_df, sample_metrics, temp_output_dir):
        """Test that JSON has required keys."""
        export_metrics_json(sample_spd_df, sample_metrics, temp_output_dir)

        output_path = os.path.join(temp_output_dir, "metrics.json")
        with open(output_path, "r") as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "summary_metrics" in data
        assert "window_level_data" in data

    def test_summary_metrics_content(
        self, sample_spd_df, sample_metrics, temp_output_dir
    ):
        """Test that summary metrics are included."""
        export_metrics_json(sample_spd_df, sample_metrics, temp_output_dir)

        output_path = os.path.join(temp_output_dir, "metrics.json")
        with open(output_path, "r") as f:
            data = json.load(f)

        summary = data["summary_metrics"]
        assert "score" in summary
        assert "win_rate" in summary
        assert "uniform_exp_decay_percentile" in summary
        assert "exp_decay_multiple_vs_uniform" in summary

    def test_window_level_data_count(
        self, sample_spd_df, sample_metrics, temp_output_dir
    ):
        """Test that window level data has correct count."""
        export_metrics_json(sample_spd_df, sample_metrics, temp_output_dir)

        output_path = os.path.join(temp_output_dir, "metrics.json")
        with open(output_path, "r") as f:
            data = json.load(f)

        assert len(data["window_level_data"]) == sample_spd_df.height

    def test_window_data_structure(
        self, sample_spd_df, sample_metrics, temp_output_dir
    ):
        """Test structure of window level data."""
        export_metrics_json(sample_spd_df, sample_metrics, temp_output_dir)

        output_path = os.path.join(temp_output_dir, "metrics.json")
        with open(output_path, "r") as f:
            data = json.load(f)

        window_data = data["window_level_data"][0]
        assert "window" in window_data
        assert "start_date" in window_data
        assert "dynamic_percentile" in window_data
        assert "uniform_percentile" in window_data
        assert "excess_percentile" in window_data


# -----------------------------------------------------------------------------
# Edge Cases
# -----------------------------------------------------------------------------


class TestExportEdgeCases:
    """Edge case tests for export_metrics_json."""

    def test_single_window(self, temp_output_dir):
        """Test export with single window."""
        single_window_df = pl.DataFrame({
            "window": ["2024-01-01 → 2025-01-01"],
            "min_sats_per_dollar": [1000],
            "max_sats_per_dollar": [5000],
            "uniform_sats_per_dollar": [2500],
            "dynamic_sats_per_dollar": [2800],
            "uniform_percentile": [37.5],
            "dynamic_percentile": [45.0],
            "excess_percentile": [7.5],
        })
        metrics = {"score": 1.0, "win_rate": 100.0}
        path = export_metrics_json(single_window_df, metrics, temp_output_dir)
        assert os.path.exists(path)

    def test_all_wins(self, temp_output_dir):
        """Test export when all windows are wins."""
        all_wins_df = pl.DataFrame({
            "window": [
                "2020-01-01 → 2021-01-01",
                "2020-02-01 → 2021-02-01",
                "2020-03-01 → 2021-03-01",
            ],
            "uniform_percentile": [30, 35, 40],
            "dynamic_percentile": [50, 55, 60],
            "min_sats_per_dollar": [1000, 1100, 900],
            "max_sats_per_dollar": [5000, 5500, 4500],
            "uniform_sats_per_dollar": [2500, 2800, 2300],
            "dynamic_sats_per_dollar": [3500, 3800, 3300],
            "excess_percentile": [20, 20, 20],
        })
        path = export_metrics_json(all_wins_df, {}, temp_output_dir)
        assert os.path.exists(path)


# -----------------------------------------------------------------------------
# Regression Tests
# -----------------------------------------------------------------------------


class TestMainRegression:
    """Regression tests for main module."""

    def test_parse_window_dates_deterministic(self):
        """Test that window date parsing is deterministic."""
        label = "2024-01-01 → 2024-12-31"

        result1 = parse_window_dates(label)
        result2 = parse_window_dates(label)

        assert result1 == result2

    def test_json_export_deterministic(
        self, sample_spd_df, sample_metrics, temp_output_dir
    ):
        """Test that JSON export produces consistent structure."""
        export_metrics_json(sample_spd_df, sample_metrics, temp_output_dir)

        output_path = os.path.join(temp_output_dir, "metrics.json")
        with open(output_path, "r") as f:
            data1 = json.load(f)

        export_metrics_json(sample_spd_df, sample_metrics, temp_output_dir)

        with open(output_path, "r") as f:
            data2 = json.load(f)

        assert data1["summary_metrics"] == data2["summary_metrics"]
        assert len(data1["window_level_data"]) == len(data2["window_level_data"])


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestMainIntegration:
    """Integration tests for main module."""

    def test_export_metrics(self, sample_spd_df, sample_metrics, temp_output_dir):
        """Test generating metrics JSON."""
        export_metrics_json(sample_spd_df, sample_metrics, temp_output_dir)

        output_path = os.path.join(temp_output_dir, "metrics.json")
        assert os.path.exists(output_path)

    def test_output_dir_created(self, sample_spd_df, temp_output_dir):
        """Test that output directory is created if it doesn't exist."""
        new_dir = os.path.join(temp_output_dir, "new_subdir")
        sample_metrics = {"score": 1.0, "win_rate": 50.0}

        export_metrics_json(sample_spd_df, sample_metrics, new_dir)

        assert os.path.exists(new_dir)
