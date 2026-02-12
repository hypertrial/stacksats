"""Tests for backtest.py - Visualization and metrics export."""

import json
import os

import pandas as pd

from stacksats.backtest import (
    create_cumulative_performance,
    create_excess_percentile_distribution,
    create_performance_comparison_chart,
    create_performance_metrics_summary,
    create_win_loss_comparison,
    export_metrics_json,
    parse_window_dates,
)

# -----------------------------------------------------------------------------
# parse_window_dates() Tests
# -----------------------------------------------------------------------------


class TestParseWindowDates:
    """Tests for the parse_window_dates function."""

    def test_parse_standard_format(self):
        """Test parsing standard window label format."""
        label = "2024-01-01 → 2024-12-31"
        result = parse_window_dates(label)

        assert isinstance(result, pd.Timestamp)
        assert result == pd.Timestamp("2024-01-01")

    def test_parse_extracts_start_date(self):
        """Test that only start date is extracted."""
        label = "2023-06-15 → 2024-06-15"
        result = parse_window_dates(label)

        assert result == pd.Timestamp("2023-06-15")

    def test_parse_different_years(self):
        """Test parsing label spanning different years."""
        label = "2020-01-01 → 2021-01-01"
        result = parse_window_dates(label)

        assert result.year == 2020
        assert result.month == 1
        assert result.day == 1


# -----------------------------------------------------------------------------
# Visualization Tests
# -----------------------------------------------------------------------------


class TestCreatePerformanceComparisonChart:
    """Tests for the create_performance_comparison_chart function."""

    def test_creates_file(self, sample_spd_df, temp_output_dir):
        """Test that chart file is created."""
        create_performance_comparison_chart(sample_spd_df, temp_output_dir)

        output_path = os.path.join(temp_output_dir, "performance_comparison.svg")
        assert os.path.exists(output_path)

    def test_file_not_empty(self, sample_spd_df, temp_output_dir):
        """Test that created file is not empty."""
        create_performance_comparison_chart(sample_spd_df, temp_output_dir)

        output_path = os.path.join(temp_output_dir, "performance_comparison.svg")
        assert os.path.getsize(output_path) > 0

    def test_no_exception(self, sample_spd_df, temp_output_dir):
        """Test that no exception is raised."""
        # Should not raise
        create_performance_comparison_chart(sample_spd_df, temp_output_dir)


class TestCreateExcessPercentileDistribution:
    """Tests for the create_excess_percentile_distribution function."""

    def test_creates_file(self, sample_spd_df, temp_output_dir):
        """Test that histogram file is created."""
        create_excess_percentile_distribution(sample_spd_df, temp_output_dir)

        output_path = os.path.join(
            temp_output_dir, "excess_percentile_distribution.svg"
        )
        assert os.path.exists(output_path)

    def test_file_not_empty(self, sample_spd_df, temp_output_dir):
        """Test that created file is not empty."""
        create_excess_percentile_distribution(sample_spd_df, temp_output_dir)

        output_path = os.path.join(
            temp_output_dir, "excess_percentile_distribution.svg"
        )
        assert os.path.getsize(output_path) > 0

    def test_no_exception(self, sample_spd_df, temp_output_dir):
        """Test that no exception is raised."""
        create_excess_percentile_distribution(sample_spd_df, temp_output_dir)


class TestCreateWinLossComparison:
    """Tests for the create_win_loss_comparison function."""

    def test_creates_file(self, sample_spd_df, temp_output_dir):
        """Test that bar chart file is created."""
        create_win_loss_comparison(sample_spd_df, temp_output_dir)

        output_path = os.path.join(temp_output_dir, "win_loss_comparison.svg")
        assert os.path.exists(output_path)

    def test_file_not_empty(self, sample_spd_df, temp_output_dir):
        """Test that created file is not empty."""
        create_win_loss_comparison(sample_spd_df, temp_output_dir)

        output_path = os.path.join(temp_output_dir, "win_loss_comparison.svg")
        assert os.path.getsize(output_path) > 0

    def test_no_exception(self, sample_spd_df, temp_output_dir):
        """Test that no exception is raised."""
        create_win_loss_comparison(sample_spd_df, temp_output_dir)


class TestCreateCumulativePerformance:
    """Tests for the create_cumulative_performance function."""

    def test_creates_file(self, sample_spd_df, temp_output_dir):
        """Test that area chart file is created."""
        create_cumulative_performance(sample_spd_df, temp_output_dir)

        output_path = os.path.join(temp_output_dir, "cumulative_performance.svg")
        assert os.path.exists(output_path)

    def test_file_not_empty(self, sample_spd_df, temp_output_dir):
        """Test that created file is not empty."""
        create_cumulative_performance(sample_spd_df, temp_output_dir)

        output_path = os.path.join(temp_output_dir, "cumulative_performance.svg")
        assert os.path.getsize(output_path) > 0

    def test_no_exception(self, sample_spd_df, temp_output_dir):
        """Test that no exception is raised."""
        create_cumulative_performance(sample_spd_df, temp_output_dir)


class TestCreatePerformanceMetricsSummary:
    """Tests for the create_performance_metrics_summary function."""

    def test_creates_file(self, sample_spd_df, sample_metrics, temp_output_dir):
        """Test that summary table file is created."""
        create_performance_metrics_summary(
            sample_spd_df, sample_metrics, temp_output_dir
        )

        output_path = os.path.join(temp_output_dir, "metrics_summary.svg")
        assert os.path.exists(output_path)

    def test_file_not_empty(self, sample_spd_df, sample_metrics, temp_output_dir):
        """Test that created file is not empty."""
        create_performance_metrics_summary(
            sample_spd_df, sample_metrics, temp_output_dir
        )

        output_path = os.path.join(temp_output_dir, "metrics_summary.svg")
        assert os.path.getsize(output_path) > 0

    def test_no_exception(self, sample_spd_df, sample_metrics, temp_output_dir):
        """Test that no exception is raised."""
        create_performance_metrics_summary(
            sample_spd_df, sample_metrics, temp_output_dir
        )


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

    def test_window_level_data_count(
        self, sample_spd_df, sample_metrics, temp_output_dir
    ):
        """Test that window level data has correct count."""
        export_metrics_json(sample_spd_df, sample_metrics, temp_output_dir)

        output_path = os.path.join(temp_output_dir, "metrics.json")
        with open(output_path, "r") as f:
            data = json.load(f)

        # Should have one entry per window
        assert len(data["window_level_data"]) == len(sample_spd_df)

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


class TestVisualizationEdgeCases:
    """Edge case tests for visualization functions."""

    def test_single_window(self, temp_output_dir):
        """Test visualizations with single window."""
        single_window_df = pd.DataFrame(
            {
                "min_sats_per_dollar": [1000],
                "max_sats_per_dollar": [5000],
                "uniform_sats_per_dollar": [2500],
                "dynamic_sats_per_dollar": [2800],
                "uniform_percentile": [37.5],
                "dynamic_percentile": [45.0],
                "excess_percentile": [7.5],
            },
            index=["2024-01-01 → 2025-01-01"],
        )

        # Should handle single window without error
        create_performance_comparison_chart(single_window_df, temp_output_dir)
        create_excess_percentile_distribution(single_window_df, temp_output_dir)
        create_win_loss_comparison(single_window_df, temp_output_dir)
        create_cumulative_performance(single_window_df, temp_output_dir)

    def test_all_wins(self, temp_output_dir):
        """Test win/loss chart when all windows are wins."""
        all_wins_df = pd.DataFrame(
            {
                "uniform_percentile": [30, 35, 40],
                "dynamic_percentile": [50, 55, 60],
                "min_sats_per_dollar": [1000, 1100, 900],
                "max_sats_per_dollar": [5000, 5500, 4500],
                "uniform_sats_per_dollar": [2500, 2800, 2300],
                "dynamic_sats_per_dollar": [3500, 3800, 3300],
                "excess_percentile": [20, 20, 20],
            },
            index=[
                "2020-01-01 → 2021-01-01",
                "2020-02-01 → 2021-02-01",
                "2020-03-01 → 2021-03-01",
            ],
        )

        create_win_loss_comparison(all_wins_df, temp_output_dir)

    def test_all_losses(self, temp_output_dir):
        """Test win/loss chart when all windows are losses."""
        all_losses_df = pd.DataFrame(
            {
                "uniform_percentile": [50, 55, 60],
                "dynamic_percentile": [30, 35, 40],
                "min_sats_per_dollar": [1000, 1100, 900],
                "max_sats_per_dollar": [5000, 5500, 4500],
                "uniform_sats_per_dollar": [2500, 2800, 2300],
                "dynamic_sats_per_dollar": [2000, 2200, 1900],
                "excess_percentile": [-20, -20, -20],
            },
            index=[
                "2020-01-01 → 2021-01-01",
                "2020-02-01 → 2021-02-01",
                "2020-03-01 → 2021-03-01",
            ],
        )

        create_win_loss_comparison(all_losses_df, temp_output_dir)


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

        # Structure should be identical (excluding timestamp)
        assert data1["summary_metrics"] == data2["summary_metrics"]
        assert len(data1["window_level_data"]) == len(data2["window_level_data"])


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestMainIntegration:
    """Integration tests for main module."""

    def test_all_visualizations(self, sample_spd_df, sample_metrics, temp_output_dir):
        """Test generating all visualizations."""
        create_performance_comparison_chart(sample_spd_df, temp_output_dir)
        create_excess_percentile_distribution(sample_spd_df, temp_output_dir)
        create_win_loss_comparison(sample_spd_df, temp_output_dir)
        create_cumulative_performance(sample_spd_df, temp_output_dir)
        create_performance_metrics_summary(
            sample_spd_df, sample_metrics, temp_output_dir
        )
        export_metrics_json(sample_spd_df, sample_metrics, temp_output_dir)

        # All files should exist
        expected_files = [
            "performance_comparison.svg",
            "excess_percentile_distribution.svg",
            "win_loss_comparison.svg",
            "cumulative_performance.svg",
            "metrics_summary.svg",
            "metrics.json",
        ]

        for filename in expected_files:
            path = os.path.join(temp_output_dir, filename)
            assert os.path.exists(path), f"Missing file: {filename}"

    def test_output_dir_created(self, sample_spd_df, temp_output_dir):
        """Test that output directory is created if it doesn't exist."""
        new_dir = os.path.join(temp_output_dir, "new_subdir")

        create_performance_comparison_chart(sample_spd_df, new_dir)

        assert os.path.exists(new_dir)
