"""Tests for visualization functions in backtest.py.

Tests chart generation functions including:
- create_performance_comparison_chart()
- create_excess_percentile_distribution()
- create_win_loss_comparison()
- create_cumulative_performance()
- create_performance_metrics_summary()
- export_metrics_json()
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stacksats.backtest import (
    create_cumulative_performance,
    create_excess_percentile_distribution,
    create_performance_comparison_chart,
    create_performance_metrics_summary,
    create_win_loss_comparison,
    export_metrics_json,
)

# -----------------------------------------------------------------------------
# Performance Comparison Chart Tests
# -----------------------------------------------------------------------------


class TestPerformanceComparisonChart:
    """Tests for create_performance_comparison_chart function."""

    def test_creates_svg_file(self, sample_spd_df, temp_output_dir):
        """Test that performance comparison chart SVG is created."""
        create_performance_comparison_chart(sample_spd_df, temp_output_dir)

        output_path = Path(temp_output_dir) / "performance_comparison.svg"
        assert output_path.exists()

    def test_svg_file_has_content(self, sample_spd_df, temp_output_dir):
        """Test that created SVG file is not empty."""
        create_performance_comparison_chart(sample_spd_df, temp_output_dir)

        output_path = Path(temp_output_dir) / "performance_comparison.svg"
        assert output_path.stat().st_size > 0

    def test_handles_single_window(self, single_window_spd_df, temp_output_dir):
        """Test chart creation with single window data."""
        create_performance_comparison_chart(single_window_spd_df, temp_output_dir)

        output_path = Path(temp_output_dir) / "performance_comparison.svg"
        assert output_path.exists()

    def test_handles_all_wins(self, all_wins_spd_df, temp_output_dir):
        """Test chart creation when dynamic always beats uniform."""
        create_performance_comparison_chart(all_wins_spd_df, temp_output_dir)

        output_path = Path(temp_output_dir) / "performance_comparison.svg"
        assert output_path.exists()


# -----------------------------------------------------------------------------
# Excess Percentile Distribution Tests
# -----------------------------------------------------------------------------


class TestExcessPercentileDistribution:
    """Tests for create_excess_percentile_distribution function."""

    def test_creates_svg_file(self, sample_spd_df, temp_output_dir):
        """Test that histogram SVG is created."""
        create_excess_percentile_distribution(sample_spd_df, temp_output_dir)

        output_path = Path(temp_output_dir) / "excess_percentile_distribution.svg"
        assert output_path.exists()

    def test_svg_file_has_content(self, sample_spd_df, temp_output_dir):
        """Test that created SVG file is not empty."""
        create_excess_percentile_distribution(sample_spd_df, temp_output_dir)

        output_path = Path(temp_output_dir) / "excess_percentile_distribution.svg"
        assert output_path.stat().st_size > 0

    def test_handles_all_positive_excess(self, all_wins_spd_df, temp_output_dir):
        """Test histogram when all excess values are positive."""
        create_excess_percentile_distribution(all_wins_spd_df, temp_output_dir)

        output_path = Path(temp_output_dir) / "excess_percentile_distribution.svg"
        assert output_path.exists()

    def test_handles_all_negative_excess(self, all_losses_spd_df, temp_output_dir):
        """Test histogram when all excess values are negative."""
        create_excess_percentile_distribution(all_losses_spd_df, temp_output_dir)

        output_path = Path(temp_output_dir) / "excess_percentile_distribution.svg"
        assert output_path.exists()


# -----------------------------------------------------------------------------
# Win/Loss Comparison Tests
# -----------------------------------------------------------------------------


class TestWinLossComparison:
    """Tests for create_win_loss_comparison function."""

    def test_creates_svg_file(self, sample_spd_df, temp_output_dir):
        """Test that bar chart SVG is created."""
        create_win_loss_comparison(sample_spd_df, temp_output_dir)

        output_path = Path(temp_output_dir) / "win_loss_comparison.svg"
        assert output_path.exists()

    def test_svg_file_has_content(self, sample_spd_df, temp_output_dir):
        """Test that created SVG file is not empty."""
        create_win_loss_comparison(sample_spd_df, temp_output_dir)

        output_path = Path(temp_output_dir) / "win_loss_comparison.svg"
        assert output_path.stat().st_size > 0

    def test_handles_100_percent_wins(self, all_wins_spd_df, temp_output_dir):
        """Test bar chart when win rate is 100%."""
        create_win_loss_comparison(all_wins_spd_df, temp_output_dir)

        output_path = Path(temp_output_dir) / "win_loss_comparison.svg"
        assert output_path.exists()

    def test_handles_0_percent_wins(self, all_losses_spd_df, temp_output_dir):
        """Test bar chart when win rate is 0%."""
        create_win_loss_comparison(all_losses_spd_df, temp_output_dir)

        output_path = Path(temp_output_dir) / "win_loss_comparison.svg"
        assert output_path.exists()


# -----------------------------------------------------------------------------
# Cumulative Performance Tests
# -----------------------------------------------------------------------------


class TestCumulativePerformance:
    """Tests for create_cumulative_performance function."""

    def test_creates_svg_file(self, sample_spd_df, temp_output_dir):
        """Test that area chart SVG is created."""
        create_cumulative_performance(sample_spd_df, temp_output_dir)

        output_path = Path(temp_output_dir) / "cumulative_performance.svg"
        assert output_path.exists()

    def test_svg_file_has_content(self, sample_spd_df, temp_output_dir):
        """Test that created SVG file is not empty."""
        create_cumulative_performance(sample_spd_df, temp_output_dir)

        output_path = Path(temp_output_dir) / "cumulative_performance.svg"
        assert output_path.stat().st_size > 0

    def test_handles_single_window(self, single_window_spd_df, temp_output_dir):
        """Test cumulative chart with single window."""
        create_cumulative_performance(single_window_spd_df, temp_output_dir)

        output_path = Path(temp_output_dir) / "cumulative_performance.svg"
        assert output_path.exists()


# -----------------------------------------------------------------------------
# Performance Metrics Summary Tests
# -----------------------------------------------------------------------------


class TestPerformanceMetricsSummary:
    """Tests for create_performance_metrics_summary function."""

    def test_creates_svg_file(self, sample_spd_df, sample_metrics, temp_output_dir):
        """Test that metrics summary table SVG is created."""
        create_performance_metrics_summary(
            sample_spd_df, sample_metrics, temp_output_dir
        )

        output_path = Path(temp_output_dir) / "metrics_summary.svg"
        assert output_path.exists()

    def test_svg_file_has_content(self, sample_spd_df, sample_metrics, temp_output_dir):
        """Test that created SVG file is not empty."""
        create_performance_metrics_summary(
            sample_spd_df, sample_metrics, temp_output_dir
        )

        output_path = Path(temp_output_dir) / "metrics_summary.svg"
        assert output_path.stat().st_size > 0


# -----------------------------------------------------------------------------
# Export Metrics JSON Tests
# -----------------------------------------------------------------------------


class TestExportMetricsJson:
    """Tests for export_metrics_json function."""

    def test_creates_json_file(self, sample_spd_df, sample_metrics, temp_output_dir):
        """Test that JSON file is created."""
        export_metrics_json(sample_spd_df, sample_metrics, temp_output_dir)

        output_path = Path(temp_output_dir) / "metrics.json"
        assert output_path.exists()

    def test_json_file_is_valid(self, sample_spd_df, sample_metrics, temp_output_dir):
        """Test that created JSON file is valid JSON."""
        export_metrics_json(sample_spd_df, sample_metrics, temp_output_dir)

        output_path = Path(temp_output_dir) / "metrics.json"
        with open(output_path) as f:
            data = json.load(f)

        assert isinstance(data, dict)

    def test_json_has_required_fields(
        self, sample_spd_df, sample_metrics, temp_output_dir
    ):
        """Test that JSON contains required top-level fields."""
        export_metrics_json(sample_spd_df, sample_metrics, temp_output_dir)

        output_path = Path(temp_output_dir) / "metrics.json"
        with open(output_path) as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "summary_metrics" in data
        assert "window_level_data" in data

    def test_json_summary_metrics(self, sample_spd_df, sample_metrics, temp_output_dir):
        """Test that summary metrics are correctly stored."""
        export_metrics_json(sample_spd_df, sample_metrics, temp_output_dir)

        output_path = Path(temp_output_dir) / "metrics.json"
        with open(output_path) as f:
            data = json.load(f)

        summary = data["summary_metrics"]

        # Check key metrics are present
        assert "score" in summary
        assert "win_rate" in summary
        assert "exp_decay_percentile" in summary
        assert "total_windows" in summary

    def test_json_window_level_data(
        self, sample_spd_df, sample_metrics, temp_output_dir
    ):
        """Test that window-level data is correctly stored."""
        export_metrics_json(sample_spd_df, sample_metrics, temp_output_dir)

        output_path = Path(temp_output_dir) / "metrics.json"
        with open(output_path) as f:
            data = json.load(f)

        window_data = data["window_level_data"]

        # Should have one entry per window
        assert len(window_data) == len(sample_spd_df)

        # Check first window has required fields
        if len(window_data) > 0:
            first_window = window_data[0]
            assert "window" in first_window
            assert "start_date" in first_window
            assert "dynamic_percentile" in first_window
            assert "uniform_percentile" in first_window
            assert "excess_percentile" in first_window

    def test_json_numeric_values_are_numbers(
        self, sample_spd_df, sample_metrics, temp_output_dir
    ):
        """Test that numeric values in JSON are proper numbers (not strings)."""
        export_metrics_json(sample_spd_df, sample_metrics, temp_output_dir)

        output_path = Path(temp_output_dir) / "metrics.json"
        with open(output_path) as f:
            data = json.load(f)

        # Check summary metrics are numbers
        summary = data["summary_metrics"]
        assert isinstance(summary["score"], (int, float))
        assert isinstance(summary["win_rate"], (int, float))

        # Check window-level data
        if len(data["window_level_data"]) > 0:
            first_window = data["window_level_data"][0]
            assert isinstance(first_window["dynamic_percentile"], (int, float))
            assert isinstance(first_window["uniform_percentile"], (int, float))


# -----------------------------------------------------------------------------
# Output Directory Tests
# -----------------------------------------------------------------------------


class TestOutputDirectory:
    """Tests for output directory handling."""

    def test_creates_output_directory_if_not_exists(
        self, sample_spd_df, temp_output_dir
    ):
        """Test that output directory is created if it doesn't exist."""
        nested_dir = Path(temp_output_dir) / "nested" / "output"

        create_performance_comparison_chart(sample_spd_df, str(nested_dir))

        assert nested_dir.exists()
        assert (nested_dir / "performance_comparison.svg").exists()

    def test_all_charts_use_same_directory(
        self, sample_spd_df, sample_metrics, temp_output_dir
    ):
        """Test that all chart functions output to same directory."""
        create_performance_comparison_chart(sample_spd_df, temp_output_dir)
        create_excess_percentile_distribution(sample_spd_df, temp_output_dir)
        create_win_loss_comparison(sample_spd_df, temp_output_dir)
        create_cumulative_performance(sample_spd_df, temp_output_dir)
        create_performance_metrics_summary(
            sample_spd_df, sample_metrics, temp_output_dir
        )
        export_metrics_json(sample_spd_df, sample_metrics, temp_output_dir)

        output_dir = Path(temp_output_dir)

        # All 6 outputs should exist
        assert (output_dir / "performance_comparison.svg").exists()
        assert (output_dir / "excess_percentile_distribution.svg").exists()
        assert (output_dir / "win_loss_comparison.svg").exists()
        assert (output_dir / "cumulative_performance.svg").exists()
        assert (output_dir / "metrics_summary.svg").exists()
        assert (output_dir / "metrics.json").exists()
