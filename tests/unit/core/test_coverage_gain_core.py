from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import stacksats.export_weights as export_weights_module
import stacksats.data.prelude as prelude_module
import stacksats.data.data_btc as data_btc_module
import stacksats.export_weights.core as export_weights_core_module
import stacksats.features.providers as feature_providers_module
import stacksats.framework_contract as framework_contract_module
import stacksats.model_development.allocation as allocation_module
import stacksats.viz.plot_weights_data as plot_weights_data_module
import stacksats.runner.helpers as runner_helpers_module
from stacksats.viz.animation_data import (
    _downsample_frame,
    _extract_window_bounds,
    _normalize_spd_frame,
    _parse_window_date,
    _select_non_overlapping_windows,
    load_backtest_payload,
    load_spd_table_from_backtest_json,
    prepare_animation_frame_data,
    spd_table_from_backtest_payload,
)
from stacksats.features.materialization import (
    build_observed_frame,
    materialize_versioned_observations,
    normalize_timestamp,
)
from stacksats.features.providers import BRKOverlayFeatureProvider, CoreModelFeatureProvider
from stacksats.features.registry import FeatureRegistry, _lazy_observed_frame
from stacksats.features.time_series import FeatureTimeSeries, _validate_no_future_data
from stacksats.model_development.allocation import (
    _normalize_intent_frame,
    _proposals_to_pl,
    _target_profile_to_pl,
)
from stacksats.model_development.weights import compute_weights_fast, compute_window_weights
from stacksats.statistical_validation import (
    _sample_blocks,
    anchored_window_excess,
    block_bootstrap_confidence_interval,
    ks_statistic,
    paired_block_permutation_pvalue,
    population_stability_index,
    whites_reality_check,
)
from stacksats.strategy_time_series import (
    StrategySeriesMetadata,
    WeightTimeSeries,
    WeightTimeSeriesBatch,
)
from stacksats.strategy_time_series.metadata import (
    _normalize_generated_at,
    _normalize_window_date,
    _parse_datetime,
)
from stacksats.strategy_time_series.schema import (
    BRK_LINEAGE,
    BRK_SOURCE_COLUMNS,
    ColumnSpec,
    brk_lineage_markdown,
    render_schema_markdown,
    validate_brk_lineage_coverage,
    validate_schema_specs,
)
from stacksats.strategy_types import (
    BaseStrategy,
    StrategyMetadata,
    _normalize_param_value,
    _to_datetime,
    _validate_metadata,
    strategy_context_from_features_df,
    validate_strategy_contract,
)


class _NoProvidersStrategy(BaseStrategy):
    strategy_id = "no-providers"

    def required_feature_sets(self) -> tuple[str, ...]:
        return ()

    def propose_weight(self, state):
        return state.uniform_weight


class _BadSignalStrategy(BaseStrategy):
    strategy_id = "bad-signal"
    intent_preference = "profile"

    def required_feature_sets(self) -> tuple[str, ...]:
        return ("core_model_features_v1",)

    def build_signals(self, ctx, features_df):
        del ctx, features_df
        return {"s": pl.Series("s", [1.0])}

    def build_target_profile(self, ctx, features_df, signals):
        del ctx, signals
        return pl.DataFrame({"date": features_df["date"], "value": [1.0] * features_df.height})


class _NonFiniteSignalStrategy(_BadSignalStrategy):
    strategy_id = "nonfinite-signal"

    def build_signals(self, ctx, features_df):
        del ctx, features_df
        return {"s": pl.Series("s", [float("inf"), 0.0])}


class _BadTargetStrategy(BaseStrategy):
    strategy_id = "bad-target"
    intent_preference = "profile"

    def required_feature_sets(self) -> tuple[str, ...]:
        return ("core_model_features_v1",)

    def build_target_profile(self, ctx, features_df, signals):
        del ctx, features_df, signals
        return "bad-target"


class _NoFeatureSetsStrategy(BaseStrategy):
    strategy_id = "no-feature-sets"

    def required_feature_sets(self) -> tuple[str, ...]:
        return ()

    def propose_weight(self, state):
        return state.uniform_weight


class _DuplicateLazyProvider:
    provider_id = "dup"

    def required_source_columns(self) -> tuple[str, ...]:
        return ("price_usd",)

    def materialize_lazy(self, btc_df, *, start_date, end_date, as_of_date):
        del btc_df, start_date, end_date, as_of_date
        return pl.DataFrame({"date": [dt.datetime(2024, 1, 1)], "dup": [1.0]}).lazy()


class _DuplicateLazyProvider2(_DuplicateLazyProvider):
    provider_id = "dup2"


class _DuplicateStrategy(BaseStrategy):
    strategy_id = "dup-strategy"

    def required_feature_sets(self) -> tuple[str, ...]:
        return ("dup", "dup2")

    def propose_weight(self, state):
        return state.uniform_weight


def _dates(n: int = 3) -> list[dt.datetime]:
    return [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]


def _feature_frame() -> pl.DataFrame:
    dates = _dates()
    return pl.DataFrame({"date": dates, "price_usd": [100.0, 101.0, 102.0], "mvrv": [1.0, 1.1, 1.2]})


def _weight_window(md: StrategySeriesMetadata | None = None) -> WeightTimeSeries:
    metadata = md or StrategySeriesMetadata(
        strategy_id="s",
        strategy_version="1.0.0",
        run_id="run-1",
        config_hash="cfg-1",
        window_start=dt.datetime(2024, 1, 1),
        window_end=dt.datetime(2024, 1, 2),
    )
    first_date = metadata.window_start or dt.datetime(2024, 1, 1)
    second_date = metadata.window_end or (first_date + dt.timedelta(days=1))
    return WeightTimeSeries(
        metadata=metadata,
        data=pl.DataFrame(
            {
                "date": [first_date, second_date],
                "weight": [0.4, 0.6],
                "price_usd": [40000.0, 41000.0],
            }
        ),
    )


def test_feature_time_series_validation_edges() -> None:
    with pytest.raises(ValueError, match="must have a 'date' column"):
        FeatureTimeSeries(pl.DataFrame({"x": [1.0]}))

    with pytest.raises(ValueError, match="must not contain nulls"):
        FeatureTimeSeries.from_dataframe(
            pl.DataFrame({"date": [dt.datetime(2024, 1, 1), None], "x": [1.0, 2.0]})
        )

    with pytest.raises(ValueError, match="must not contain duplicates"):
        FeatureTimeSeries.from_dataframe(
            pl.DataFrame({"date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 1)], "x": [1.0, 2.0]})
        )

    with pytest.raises(ValueError, match="sorted ascending"):
        FeatureTimeSeries.from_dataframe(
            pl.DataFrame({"date": [dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 1)], "x": [1.0, 2.0]})
        )

    with pytest.raises(ValueError, match="after as_of_date"):
        FeatureTimeSeries.from_dataframe(
            pl.DataFrame({"date": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 3)], "x": [1.0, 2.0]}),
            as_of_date=dt.datetime(2024, 1, 2),
        )

    with pytest.raises(ValueError, match="must not contain nulls"):
        FeatureTimeSeries.from_dataframe(
            pl.DataFrame({"date": _dates(2), "x": [1.0, None]}),
            require_finite=("x",),
        )

    with pytest.raises(ValueError, match="no inf"):
        FeatureTimeSeries.from_dataframe(
            pl.DataFrame({"date": _dates(2), "x": [1.0, float("inf")]}),
            require_finite=("x",),
        )

    empty = FeatureTimeSeries.from_dataframe(pl.DataFrame(schema={"date": pl.Datetime("us")}))
    assert empty.row_count == 0

    series = FeatureTimeSeries.from_dataframe(pl.DataFrame({"date": _dates(2), "x": [1.0, 2.0]}))
    assert series.columns == ("date", "x")
    with pytest.raises(ValueError, match="missing required column"):
        series.validate_schema(("missing",))


def test_feature_time_series_helper_edges() -> None:
    with pytest.raises(TypeError, match="must be a polars DataFrame"):
        FeatureTimeSeries(_frame=object())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="requires a polars DataFrame"):
        FeatureTimeSeries.from_dataframe(object())  # type: ignore[arg-type]

    renamed = FeatureTimeSeries.from_dataframe(
        pl.DataFrame({"when": _dates(2), "signal": [1.0, 2.0]})
    )
    assert renamed.columns == ("date", "signal")

    with pytest.raises(ValueError, match="missing required column"):
        FeatureTimeSeries.from_dataframe(
            pl.DataFrame({"when": _dates(1)}),
            required_columns=("signal",),
        )

    FeatureTimeSeries.from_dataframe(
        pl.DataFrame({"date": _dates(2), "x": [1, 2]}),
        require_finite=("missing",),
    )

    series = FeatureTimeSeries.from_dataframe(pl.DataFrame({"date": _dates(2), "x": [1.0, 2.0]}))
    series.validate_schema(("date", "x"))

    FeatureTimeSeries.from_dataframe(
        pl.DataFrame({"date": _dates(2), "x": [1.0, 2.0]}),
        require_finite=("x",),
    )

    FeatureTimeSeries.from_dataframe(
        pl.DataFrame({"date": _dates(2), "signal": [1.0, 2.0]}),
        require_finite=("missing",),
    )

    _validate_no_future_data(pl.DataFrame(schema={"date": pl.Datetime("us")}), dt.datetime(2024, 1, 1))
    with pytest.raises(ValueError, match="must not contain data after as_of_date"):
        _validate_no_future_data(
            pl.DataFrame({"date": ["2024-01-03"]}),
            dt.datetime(2024, 1, 2, 12, tzinfo=dt.timezone.utc),
        )


def test_metadata_and_batch_error_edges(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="Expected datetime or str"):
        StrategySeriesMetadata(
            strategy_id="s",
            strategy_version="1",
            run_id="r",
            config_hash="c",
            generated_at=object(),  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="window_start must be <= window_end"):
        StrategySeriesMetadata(
            strategy_id="s",
            strategy_version="1",
            run_id="r",
            config_hash="c",
            window_start=dt.datetime(2024, 1, 2),
            window_end=dt.datetime(2024, 1, 1),
        )

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Artifact metadata file not found"):
        WeightTimeSeriesBatch.from_artifact_dir(artifact_dir)

    bad_json = artifact_dir / "artifacts.json"
    bad_json.write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid artifacts.json"):
        WeightTimeSeriesBatch.from_artifact_dir(artifact_dir)

    bad_json.write_text(json.dumps({"strategy_id": "s"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required provenance fields"):
        WeightTimeSeriesBatch.from_artifact_dir(artifact_dir)

    bad_json.write_text(
        json.dumps(
            {
                "strategy_id": "s",
                "version": "1",
                "run_id": "r",
                "config_hash": "c",
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="field 'files' must be an object"):
        WeightTimeSeriesBatch.from_artifact_dir(artifact_dir)

    bad_json.write_text(
        json.dumps(
            {
                "strategy_id": "s",
                "version": "1",
                "run_id": "r",
                "config_hash": "c",
                "files": {"weights_csv": ""},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-empty 'weights_csv'"):
        WeightTimeSeriesBatch.from_artifact_dir(artifact_dir)

    bad_json.write_text(
        json.dumps(
            {
                "strategy_id": "s",
                "version": "1",
                "run_id": "r",
                "config_hash": "c",
                "files": {"weights_csv": "missing.csv"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="weights.csv not found"):
        WeightTimeSeriesBatch.from_artifact_dir(artifact_dir)

    base_meta = StrategySeriesMetadata(
        strategy_id="s",
        strategy_version="1",
        run_id="r",
        config_hash="c",
        window_start=dt.datetime(2024, 1, 1),
        window_end=dt.datetime(2024, 1, 2),
    )
    with pytest.raises(ValueError, match="strategy_version does not match"):
        WeightTimeSeriesBatch(
            strategy_id="s",
            strategy_version="2",
            run_id="r",
            config_hash="c",
            windows=(_weight_window(base_meta),),
        )

    mismatched_meta = StrategySeriesMetadata(
        strategy_id="s",
        strategy_version="1",
        run_id="r",
        config_hash="c",
        generated_at=dt.datetime(2025, 1, 2, tzinfo=dt.timezone.utc),
        window_start=dt.datetime(2024, 1, 3),
        window_end=dt.datetime(2024, 1, 4),
    )
    with pytest.raises(ValueError, match="generated_at does not match"):
        WeightTimeSeriesBatch(
            strategy_id="s",
            strategy_version="1",
            run_id="r",
            config_hash="c",
            windows=(_weight_window(base_meta), _weight_window(mismatched_meta)),
        )


def test_metadata_and_schema_helper_edges() -> None:
    assert _parse_datetime("2024-01-01") == dt.datetime(2024, 1, 1)
    assert _normalize_generated_at(dt.datetime(2024, 1, 1)).tzinfo == dt.timezone.utc
    assert _normalize_window_date("2024-01-01T12:00:00+01:00") == dt.datetime(2024, 1, 1)

    invalid_specs = [
        dict(name="", dtype="float64", required=True, description="desc"),
        dict(name="x", dtype="", required=True, description="desc"),
        dict(name="x", dtype="float64", required="yes", description="desc"),
        dict(name="x", dtype="float64", required=True, description=""),
        dict(name="x", dtype="float64", required=True, description="desc", unit=1),
        dict(name="x", dtype="float64", required=True, description="desc", constraints=["a"]),
        dict(name="x", dtype="float64", required=True, description="desc", constraints=(" ",)),
        dict(name="x", dtype="float64", required=True, description="desc", source=""),
        dict(name="x", dtype="float64", required=True, description="desc", formula=1),
    ]
    for kwargs in invalid_specs:
        with pytest.raises((TypeError, ValueError)):
            ColumnSpec(**kwargs)  # type: ignore[arg-type]

    extra = ColumnSpec(name="extra_metric", dtype="float64", required=False, description="extra | metric")
    assert "extra \\| metric" in render_schema_markdown((extra,))
    assert "source_column" in brk_lineage_markdown(BRK_LINEAGE[:1])

    dup = ColumnSpec(name="dup", dtype="float64", required=False, description="dup")
    with pytest.raises(ValueError, match="duplicate column names"):
        validate_schema_specs((dup, dup))
    with pytest.raises(ValueError, match="collide with core WeightTimeSeries schema"):
        validate_schema_specs((ColumnSpec(name="date", dtype="datetime64[ns]", required=False, description="dup"),))
    with pytest.raises(ValueError, match="reference undocumented WeightTimeSeries columns"):
        validate_brk_lineage_coverage(
            lineage=(BRK_LINEAGE[0],),
            schema_specs_iter=(extra,),
            source_columns=(BRK_LINEAGE[0].source_column,),
        )
    with pytest.raises(ValueError, match="missing source columns"):
        validate_brk_lineage_coverage(
            lineage=(BRK_LINEAGE[0],),
            schema_specs_iter=(ColumnSpec(name="date", dtype="datetime64[ns]", required=True, description="date"),),
            source_columns=BRK_SOURCE_COLUMNS[:2],
        )


def test_statistical_validation_edges() -> None:
    assert anchored_window_excess(pl.DataFrame({"x": [1.0]}), step=2).size == 0
    assert anchored_window_excess(
        pl.DataFrame({"excess_percentile": [float("nan"), float("inf")]}),
        step=2,
    ).size == 0

    interval = block_bootstrap_confidence_interval(np.array([]), block_size=2, trials=5, seed=1)
    assert interval.lower == 0.0 and interval.upper == 0.0 and interval.samples.size == 0
    sampled_interval = block_bootstrap_confidence_interval(
        np.array([1.0, 2.0, 3.0]),
        block_size=10,
        trials=4,
        seed=1,
    )
    assert sampled_interval.samples.size == 4
    assert sampled_interval.lower <= sampled_interval.upper

    assert paired_block_permutation_pvalue(np.array([]), np.array([]), block_size=2, trials=5, seed=1) == 1.0
    assert population_stability_index(np.array([]), np.array([1.0])) == 0.0
    assert population_stability_index(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.0
    assert population_stability_index(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])) == 0.0
    assert ks_statistic(np.array([]), np.array([1.0])) == 0.0
    assert whites_reality_check({"a": np.array([np.nan, np.nan])}, block_size=2, trials=5, seed=1) == 1.0
    assert np.array_equal(
        _sample_blocks(np.array([1.0, 2.0]), 5, np.random.default_rng(1)),
        np.array([1.0, 2.0]),
    )


def test_animation_data_edges(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_backtest_payload(tmp_path / "missing.json")

    non_object = tmp_path / "bad_root.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be an object"):
        load_backtest_payload(non_object)

    with pytest.raises(ValueError, match="missing 'window_level_data' list"):
        spd_table_from_backtest_payload({})
    with pytest.raises(ValueError, match="empty 'window_level_data'"):
        spd_table_from_backtest_payload({"window_level_data": []})
    with pytest.raises(ValueError, match="could not be parsed"):
        spd_table_from_backtest_payload({"window_level_data": [[]]})
    with pytest.raises(ValueError, match="missing required columns"):
        prepare_animation_frame_data(pl.DataFrame({"window": ["2024-01-01 → 2024-01-02"]}))

    with pytest.raises(ValueError, match="Invalid date"):
        _parse_window_date("bad-date")
    assert _parse_window_date("2024-01-01") == dt.datetime(2024, 1, 1)
    with pytest.raises(ValueError, match="Invalid window label"):
        _extract_window_bounds("2024-01-01")
    with pytest.raises(ValueError, match="Invalid window label"):
        _extract_window_bounds("2024-01-01 → ")
    start, end = _extract_window_bounds("2024-01-01 → 2024-12-31")
    assert start == dt.datetime(2024, 1, 1)
    assert end == dt.datetime(2024, 12, 31)
    with pytest.raises(ValueError, match="window' or 'index'"):
        _normalize_spd_frame(pl.DataFrame({"x": [1.0]}))
    with pytest.raises(ValueError, match="no valid numeric values"):
        _normalize_spd_frame(
            pl.DataFrame(
                {
                    "window": ["2024-01-01 → 2024-01-02"],
                    "dynamic_percentile": ["bad"],
                    "uniform_percentile": [1.0],
                    "excess_percentile": [0.0],
                    "dynamic_sats_per_dollar": [10.0],
                    "uniform_sats_per_dollar": [10.0],
                }
            )
        )

    flat = pl.DataFrame(
        {
            "window": ["2024-01-01 → 2024-01-02"],
            "dynamic_percentile": [1.0],
            "uniform_percentile": [1.0],
            "excess_percentile": [0.0],
            "dynamic_sats_per_dollar": [10.0],
            "uniform_sats_per_dollar": [10.0],
        }
    )
    with pytest.raises(ValueError, match="max_frames must be > 0"):
        prepare_animation_frame_data(flat, max_frames=0)
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps({"window_level_data": flat.to_dicts()}), encoding="utf-8")
    loaded = load_spd_table_from_backtest_json(payload)
    assert loaded.columns == flat.columns
    empty_selected = _select_non_overlapping_windows(
        pl.DataFrame(
            schema={
                "window": pl.Utf8,
                "dynamic_percentile": pl.Float64,
                "uniform_percentile": pl.Float64,
                "excess_percentile": pl.Float64,
                "dynamic_sats_per_dollar": pl.Float64,
                "uniform_sats_per_dollar": pl.Float64,
                "window_start": pl.Datetime("us"),
                "window_end": pl.Datetime("us"),
            }
        )
    )
    assert empty_selected.is_empty()
    downsampled = _downsample_frame(
        pl.DataFrame({"x": [1, 2, 3, 4], "y": [1, 2, 3, 4]}),
        max_frames=1,
    )
    assert downsampled.height == 2
    prepared = prepare_animation_frame_data(flat)
    assert float(prepared["cumulative_btc_vs_uniform_pct"][0]) == 0.0
    empty_flat = pl.DataFrame(
        schema={
            "window": pl.Utf8,
            "dynamic_percentile": pl.Float64,
            "uniform_percentile": pl.Float64,
            "excess_percentile": pl.Float64,
            "dynamic_sats_per_dollar": pl.Float64,
            "uniform_sats_per_dollar": pl.Float64,
        }
    )
    with pytest.raises(ValueError, match="no valid numeric values"):
        prepare_animation_frame_data(empty_flat)


def test_data_btc_helper_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(data_btc_module.DataLoadError, match="must have a 'date' column"):
        data_btc_module._require_daily_index(
            pl.DataFrame({"x": [1.0]}),
            backtest_start_ts=dt.datetime(2024, 1, 1),
            target_end=dt.datetime(2024, 1, 2),
        )
    with pytest.raises(data_btc_module.DataLoadError, match="must not contain nulls"):
        data_btc_module._require_daily_index(
            pl.DataFrame({"date": [dt.datetime(2024, 1, 1), None]}),
            backtest_start_ts=dt.datetime(2024, 1, 1),
            target_end=dt.datetime(2024, 1, 2),
        )
    with pytest.raises(data_btc_module.DataLoadError, match="No valid dates"):
        data_btc_module._require_daily_index(
            pl.DataFrame(schema={"date": pl.Datetime("us")}),
            backtest_start_ts=dt.datetime(2024, 1, 1),
            target_end=dt.datetime(2024, 1, 2),
        )
    with pytest.raises(data_btc_module.DataLoadError, match="does not cover requested window"):
        data_btc_module._require_daily_index(
            pl.DataFrame({"date": _dates(2)}),
            backtest_start_ts=dt.datetime(2024, 1, 1),
            target_end=dt.datetime(2024, 1, 3),
        )

    renamed_path = tmp_path / "renamed.parquet"
    pl.DataFrame({"timestamp": _dates(2), "price_usd": [1.0, 2.0]}).write_parquet(renamed_path)
    renamed = data_btc_module._load_btc_from_parquet(renamed_path)
    assert renamed.columns[0] == "date"

    bad_path = tmp_path / "bad.parquet"
    pl.DataFrame({"x": [1.0], "price_usd": [1.0]}).write_parquet(bad_path)
    with pytest.raises(data_btc_module.DataLoadError, match="must have a 'date' column"):
        data_btc_module._load_btc_from_parquet(bad_path)

    monkeypatch.setattr(
        data_btc_module,
        "_resolve_parquet_path",
        lambda _path: Path("unused.parquet"),
    )

    monkeypatch.setattr(
        data_btc_module,
        "_scan_btc_from_parquet",
        lambda _path: pl.DataFrame({"date": _dates(2)}).lazy(),
    )
    with pytest.raises(data_btc_module.DataLoadError, match="Required price_usd series missing"):
        data_btc_module.BTCDataProvider(clock=lambda: dt.datetime(2024, 1, 2)).load(
            backtest_start="2024-01-01",
            end_date="2024-01-02",
        )

    monkeypatch.setattr(
        data_btc_module,
        "_scan_btc_from_parquet",
        lambda _path: pl.DataFrame(
            {"date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)], "price_usd": [1.0, 2.0]}
        ).lazy(),
    )
    loaded = data_btc_module.BTCDataProvider(clock=lambda: dt.datetime(2024, 1, 2)).load(
        backtest_start="2024-01-01",
        end_date="2024-01-02",
    )
    assert loaded.height == 2

    monkeypatch.setattr(
        data_btc_module,
        "_scan_btc_from_parquet",
        lambda _path: pl.DataFrame({"date": [dt.datetime(2024, 1, 3)], "price_usd": [2.0]}).lazy(),
    )
    with pytest.raises(data_btc_module.DataLoadError, match="No BRK rows available"):
        data_btc_module.BTCDataProvider(clock=lambda: dt.datetime(2024, 1, 3)).load(
            backtest_start="2024-01-02",
            end_date="2024-01-02",
        )


def test_feature_materialization_and_registry_edges() -> None:
    with pytest.raises(ValueError, match="must have 'date' column"):
        build_observed_frame(pl.DataFrame({"x": [1.0]}), start_date="2024-01-01", current_date="2024-01-02")

    observed = build_observed_frame(
        pl.DataFrame({"date": _dates(2), "x": [1.0, 2.0]}),
        start_date="2024-01-03",
        current_date="2024-01-02",
    )
    assert observed.is_empty()

    assert materialize_versioned_observations(pl.DataFrame(), as_of_date="2024-01-01").is_empty()
    obs = pl.DataFrame(
        {
            "effective_date": ["2024-01-01"],
            "available_at": ["2024-01-03"],
            "signal": [1.0],
        }
    )
    assert materialize_versioned_observations(obs, as_of_date="2024-01-01").is_empty()
    assert normalize_timestamp(dt.datetime(2024, 1, 1, 12, tzinfo=dt.timezone.utc)) == dt.datetime(2024, 1, 1)

    with pytest.raises(ValueError, match="must have 'date' column"):
        _lazy_observed_frame(
            pl.DataFrame({"x": [1.0]}).lazy(),
            schema=pl.Schema({"x": pl.Float64}),
            start_date=dt.datetime(2024, 1, 1),
            current_date=dt.datetime(2024, 1, 2),
        ).collect()

    registry = FeatureRegistry()
    registry.register(_DuplicateLazyProvider())
    registry.register(_DuplicateLazyProvider2())

    with pytest.raises(ValueError, match="duplicate columns"):
        registry.materialize_for_strategy(
            _DuplicateStrategy(),
            pl.DataFrame({"date": _dates(2), "price_usd": [1.0, 2.0]}),
            start_date=dt.datetime(2024, 1, 1),
            end_date=dt.datetime(2024, 1, 2),
            current_date=dt.datetime(2024, 1, 2),
        )

    frame = FeatureRegistry().materialize_for_strategy(
        _NoProvidersStrategy(),
        pl.DataFrame({"date": _dates(2), "price_usd": [1.0, 2.0]}),
        start_date=dt.datetime(2024, 1, 1),
        end_date=dt.datetime(2024, 1, 2),
        current_date=dt.datetime(2024, 1, 2),
    )
    assert frame.columns == ["date"]


def test_framework_runner_registry_and_provider_helper_edges() -> None:
    class _PyDate:
        def to_pydatetime(self):
            return dt.datetime(2024, 1, 2, 14)

    class _FallbackDate:
        def __str__(self):
            return "2024-01-03 trailing"

    assert framework_contract_module._as_timestamp("2024-01-01 trailing") == dt.datetime(2024, 1, 1)
    assert framework_contract_module._as_timestamp(
        dt.datetime(2024, 1, 1, 12, tzinfo=dt.timezone.utc)
    ) == dt.datetime(2024, 1, 1)
    assert framework_contract_module._day_count(dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 1)) == 0
    assert framework_contract_module._to_naive_utc(_PyDate()) == dt.datetime(2024, 1, 2)
    assert framework_contract_module._to_naive_utc(_FallbackDate()) == dt.datetime(2024, 1, 3)

    assert runner_helpers_module._value_col(pl.DataFrame()) == ""
    assert runner_helpers_module.weights_match(pl.DataFrame(), pl.DataFrame()) is True
    assert runner_helpers_module.weights_match(pl.DataFrame(), pl.DataFrame({"date": _dates(1), "weight": [1.0]})) is False

    observed_utf8 = _lazy_observed_frame(
        pl.DataFrame({"date": ["2024-01-01", "2024-01-02"], "x": [1.0, 2.0]}).lazy(),
        schema=pl.Schema({"date": pl.Utf8, "x": pl.Float64}),
        start_date=dt.datetime(2024, 1, 1),
        current_date=dt.datetime(2024, 1, 2),
    ).collect()
    assert observed_utf8.height == 2
    observed_date = _lazy_observed_frame(
        pl.DataFrame({"date": [dt.date(2024, 1, 1)], "x": [1.0]}).lazy(),
        schema=pl.Schema({"date": pl.Date, "x": pl.Float64}),
        start_date=dt.datetime(2024, 1, 1),
        current_date=dt.datetime(2024, 1, 1),
    ).collect()
    assert observed_date["date"][0] == dt.datetime(2024, 1, 1)

    class _EagerOnlyProvider:
        provider_id = "eager-only"

        def required_source_columns(self) -> tuple[str, ...]:
            return ("price_usd",)

        def materialize(self, btc_df, *, start_date, end_date, as_of_date):
            del btc_df, start_date, end_date, as_of_date
            return pl.DataFrame({"date": _dates(1), "eager_signal": [1.0]})

    class _EagerStrategy(BaseStrategy):
        strategy_id = "eager-strategy"

        def required_feature_sets(self) -> tuple[str, ...]:
            return ("eager-only",)

        def propose_weight(self, state):
            return state.uniform_weight

    eager_registry = FeatureRegistry()
    eager_registry.register(_EagerOnlyProvider())
    eager_frame = eager_registry.materialize_for_strategy(
        _EagerStrategy(),
        pl.DataFrame({"date": _dates(1), "price_usd": [1.0]}),
        start_date=dt.datetime(2024, 1, 1),
        end_date=dt.datetime(2024, 1, 1),
        current_date=dt.datetime(2024, 1, 1),
    )
    assert eager_frame.columns == ["date", "eager_signal"]

    assert runner_helpers_module.build_fold_ranges(
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 6, 1),
    ) == []

    empty = pl.DataFrame(schema={"date": pl.Datetime("us"), "price_usd": pl.Float64, "mvrv": pl.Float64})
    assert runner_helpers_module.build_window_index(pl.DataFrame({"date": _dates(1)}))[1]
    assert CoreModelFeatureProvider().materialize_lazy(
        empty,
        start_date=dt.datetime(2024, 1, 1),
        end_date=dt.datetime(2024, 1, 2),
        as_of_date=dt.datetime(2024, 1, 2),
    ).collect().is_empty()
    assert BRKOverlayFeatureProvider().materialize(
        empty,
        start_date=dt.datetime(2024, 1, 1),
        end_date=dt.datetime(2024, 1, 2),
        as_of_date=dt.datetime(2024, 1, 2),
    ).is_empty()
    assert BRKOverlayFeatureProvider()._cache_key is None

    overlay_provider = BRKOverlayFeatureProvider()
    overlay_source = pl.DataFrame(
        {
            "date": _dates(40),
            "price_usd": [100.0 + i for i in range(40)],
            "mvrv": [1.0 + (i / 100.0) for i in range(40)],
        }
    )
    first = overlay_provider.materialize(
        overlay_source,
        start_date=dt.datetime(2024, 1, 1),
        end_date=dt.datetime(2024, 2, 9),
        as_of_date=dt.datetime(2024, 2, 9),
    )
    second = overlay_provider.materialize(
        overlay_source,
        start_date=dt.datetime(2024, 1, 1),
        end_date=dt.datetime(2024, 2, 9),
        as_of_date=dt.datetime(2024, 2, 9),
    )
    assert first.equals(second)


def test_feature_provider_empty_branches() -> None:
    empty = pl.DataFrame(schema={"date": pl.Datetime("us"), "price_usd": pl.Float64})
    assert CoreModelFeatureProvider().materialize(
        empty,
        start_date=dt.datetime(2024, 1, 1),
        end_date=dt.datetime(2024, 1, 2),
        as_of_date=dt.datetime(2024, 1, 2),
    ).is_empty()
    assert BRKOverlayFeatureProvider().materialize_lazy(
        empty,
        start_date=dt.datetime(2024, 1, 1),
        end_date=dt.datetime(2024, 1, 2),
        as_of_date=dt.datetime(2024, 1, 2),
    ).collect().is_empty()


def test_prelude_and_strategy_type_edges() -> None:
    frame = pl.DataFrame({"date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)], "price_usd": [10.0, 10.0]})
    result = prelude_module.compute_cycle_spd(
        frame,
        strategy_function=lambda window: pl.DataFrame({"date": window["date"][:1], "weight": [1.0]}),
        features_df=frame,
        start_date="2024-01-01",
        end_date="2025-01-01",
        validate_weights=False,
    )
    assert result.is_empty()

    with pytest.raises(ValueError, match="must have 'date' column"):
        prelude_module._ensure_pl_with_date(pl.DataFrame({"x": [1.0]}))
    assert len(prelude_module.date_range_list("2024-01-01", "2024-01-02")) == 2
    assert prelude_module.date_range_series("2024-01-01", "2024-01-02").len() == 2

    class _PyDate:
        def to_pydatetime(self):
            return dt.datetime(2024, 1, 1, 15)

    assert _to_datetime(_PyDate()) == dt.datetime(2024, 1, 1)
    assert _to_datetime("2024-01-01T12:00:00Z") == dt.datetime(2024, 1, 1)
    assert _normalize_param_value(Path("foo"), key_path="x") == "foo"

    class _Iso:
        def isoformat(self):
            return "2024-01-01T00:00:00"

    assert _normalize_param_value(_Iso(), key_path="x") == "2024-01-01T00:00:00"
    with pytest.raises(TypeError, match="dict keys must be strings"):
        _normalize_param_value({1: "x"}, key_path="x")
    with pytest.raises(TypeError, match="Unsupported strategy param value"):
        _normalize_param_value(object(), key_path="x")
    with pytest.raises(TypeError, match="description must be a string"):
        _validate_metadata(StrategyMetadata("id", "1", description=1))  # type: ignore[arg-type]

    features_df = _feature_frame().slice(0, 2)
    with pytest.raises(ValueError, match="signal 's' must be pl.Series with length matching window"):
        _BadSignalStrategy().compute_weights(
            strategy_context_from_features_df(features_df, "2024-01-01", "2024-01-02", "2024-01-02")
        )


def test_export_core_and_prelude_helper_edges() -> None:
    assert export_weights_core_module._normalize_date_frame(pl.DataFrame({"x": [1.0]})).columns == ["x"]
    normalized_utf8 = export_weights_core_module._normalize_date_frame(
        pl.DataFrame({"date": ["2024-01-01"]})
    )
    assert normalized_utf8["date"][0] == dt.datetime(2024, 1, 1)
    normalized_date = export_weights_core_module._normalize_date_frame(
        pl.DataFrame({"date": [dt.date(2024, 1, 2)]})
    )
    assert normalized_date["date"][0] == dt.datetime(2024, 1, 2)

    captured_locked: list[np.ndarray | None] = []
    exported = export_weights_core_module.process_start_date_batch(
        "2024-01-01",
        ["2024-01-01"],
        pl.DataFrame({"date": [dt.datetime(2024, 1, 1)], "price_usd": [100.0]}),
        pl.DataFrame({"date": [dt.datetime(2024, 1, 1)], "price_usd": [100.0]}),
        "2024-01-01",
        "price_usd",
        locked_weights_by_end_date={"2024-01-01": np.array([0.25])},
        enforce_span_contract=False,
        compute_window_weights_fn=lambda *_args, locked_weights=None, **_kwargs: (
            captured_locked.append(locked_weights)
            or pl.DataFrame({"date": [dt.datetime(2024, 1, 1)], "weight": [1.0]})
        ),
        validate_span_length_fn=lambda *_args, **_kwargs: None,
        base_strategy_cls=BaseStrategy,
        validate_strategy_contract_fn=lambda *_args, **_kwargs: None,
    )
    assert captured_locked and np.array_equal(captured_locked[0], np.array([0.25]))
    assert exported.height == 1

    utf8_frame = pl.DataFrame({"date": ["2024-01-01"], "price_usd": [10.0]})
    normalized = utf8_frame.with_columns(prelude_module._daily_datetime_expr(utf8_frame).alias("date"))
    assert normalized["date"][0] == dt.datetime(2024, 1, 1)

    price_slice = pl.DataFrame({"date": _dates(3), "price_usd": [10.0, 11.0, 12.0]})
    aligned = prelude_module._normalize_weight_frame(
        price_slice,
        pl.DataFrame({"date": _dates(2), "weight": [0.2, 0.3]}),
    )
    assert aligned.height == price_slice.height
    assert pytest.approx(float(aligned["weight"].sum())) == 1.0

    window_days = framework_contract_module.ALLOCATION_SPAN_DAYS
    constant_dates = _dates(window_days)
    constant = pl.DataFrame({"date": constant_dates, "price_usd": [10.0] * window_days})
    spd = prelude_module.compute_cycle_spd(
        constant,
        strategy_function=lambda window: pl.DataFrame(
            {
                "date": window["date"],
                "weight": [1.0 / window.height] * window.height,
            }
        ),
        features_df=constant,
        start_date="2024-01-01",
        end_date=constant_dates[-1].strftime("%Y-%m-%d"),
        validate_weights=False,
    )
    assert spd.height == 1
    assert str(spd["uniform_percentile"][0]) == "nan"
    assert str(spd["dynamic_percentile"][0]) == "nan"


def test_export_weight_price_fallback_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    fallback_calls: list[float | None] = []

    def _fallback(*, previous_price=None, fetch_btc_price_fn=None):
        del fetch_btc_price_fn
        fallback_calls.append(previous_price)
        return 111.0

    monkeypatch.setattr(export_weights_module, "_get_current_btc_price", _fallback)

    class _MissingPriceProvider:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def load(self, **kwargs):
            del kwargs
            return pl.DataFrame({"date": [dt.datetime(2024, 1, 1)]})

    monkeypatch.setattr(export_weights_module, "BTCDataProvider", _MissingPriceProvider)
    assert export_weights_module.get_current_btc_price(previous_price=10.0) == 111.0
    assert fallback_calls == [10.0]

    today = dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_str = (today - dt.timedelta(days=1)).strftime("%Y-%m-%d")

    class _StringDateProvider:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def load(self, **kwargs):
            del kwargs
            return pl.DataFrame(
                {
                    "date": [yesterday_str, "2000-01-01"],
                    "price_usd": [float("nan"), 123.0],
                }
            )

    monkeypatch.setattr(export_weights_module, "BTCDataProvider", _StringDateProvider)
    assert export_weights_module.get_current_btc_price() == 123.0

    class _DatetimeProvider:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def load(self, **kwargs):
            del kwargs
            return pl.DataFrame(
                {
                    "date": [today, today - dt.timedelta(days=1)],
                    "price_usd": [456.0, 400.0],
                }
            )

    monkeypatch.setattr(export_weights_module, "BTCDataProvider", _DatetimeProvider)
    assert export_weights_module.get_current_btc_price() == 456.0


def test_remaining_helper_branch_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    assert feature_providers_module._btc_frame_cache_key(
        pl.DataFrame(),
        source_columns=("price_usd",),
    ) == (0, "")

    monkeypatch.setattr(
        plot_weights_data_module,
        "get_date_range_options",
        lambda _conn: pl.DataFrame({"start_date": ["2024-01-01"], "end_date": ["2024-01-02"]}),
    )
    assert plot_weights_data_module.get_oldest_date_range(object()) == ("2024-01-01", "2024-01-02")

    assert _target_profile_to_pl(type("_Profile", (), {"values": pl.DataFrame({"date": _dates(1), "value": [1.0]})})()).height == 1
    assert feature_providers_module._btc_frame_cache_key(
        pl.DataFrame({"date": _dates(1), "price_usd": [1.0]}),
        source_columns=("price_usd",),
    )[0] == 1
    assert feature_providers_module.DATE_COL == "date"
    assert plot_weights_data_module.get_oldest_date_range(object()) == ("2024-01-01", "2024-01-02")
    assert allocation_module._to_datetime("2024-01-01 extra") == dt.datetime(2024, 1, 1)
    assert allocation_module._to_datetime(
        dt.datetime(2024, 1, 2, 9, tzinfo=dt.timezone.utc)
    ) == dt.datetime(2024, 1, 2)


def test_strategy_type_and_model_weight_edges() -> None:
    class _LintFailStrategy(BaseStrategy):
        strategy_id = "lint-fail"

        def required_feature_sets(self) -> tuple[str, ...]:
            return ("missing-provider",)

        def propose_weight(self, state):
            return state.uniform_weight

    strat = _NoFeatureSetsStrategy()
    with pytest.raises(ValueError, match="at least one required_feature_set"):
        validate_strategy_contract(strat)

    bad_pref = _NoProvidersStrategy()
    bad_pref.intent_preference = "bad"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="intent_preference"):
        validate_strategy_contract(bad_pref)

    with pytest.raises(KeyError, match="Unknown feature provider"):
        validate_strategy_contract(_LintFailStrategy())

    with pytest.raises(TypeError, match="target_profile must be a Polars DataFrame"):
        _target_profile_to_pl(object())
    with pytest.raises(TypeError, match="proposals must be a Polars DataFrame"):
        _proposals_to_pl(object())
    with pytest.raises(TypeError, match="must be a Polars DataFrame"):
        _normalize_intent_frame([], name="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must have 'date' and 'value'"):
        _normalize_intent_frame(pl.DataFrame({"date": _dates(1)}), name="bad")
    assert _normalize_intent_frame(pl.DataFrame({"date": [], "value": []}), name="empty").is_empty()
    assert feature_providers_module.DATE_COL == "date"
    assert compute_window_weights.__name__ == "compute_window_weights"
    assert framework_contract_module._to_naive_utc(dt.datetime(2024, 1, 1, 12, tzinfo=dt.timezone.utc)) == dt.datetime(2024, 1, 1)
    assert framework_contract_module._to_naive_utc(type("_Py", (), {"to_pydatetime": lambda self: dt.datetime(2024, 1, 2, 8)})()) == dt.datetime(2024, 1, 2)
    assert framework_contract_module._to_naive_utc(type("_Str", (), {"__str__": lambda self: "2024-01-03 suffix"})()) == dt.datetime(2024, 1, 3)

    empty_fast = compute_weights_fast(
        pl.DataFrame({"date": _dates(1), "price_usd": [1.0]}),
        "2030-01-01",
        "2030-01-02",
        compute_preference_scores_fn=lambda *_args, **_kwargs: pl.DataFrame({"date": [], "preference": []}),
        allocate_sequential_stable_fn=lambda raw, n_past, locked: raw,
    )
    assert empty_fast.is_empty()

    with pytest.raises(ValueError, match="preference score frame must have 'date' and 'preference' columns"):
        compute_window_weights(
            _feature_frame(),
            "2024-01-01",
            "2024-01-03",
            "2024-01-03",
            validate_span_length_fn=lambda *_args, **_kwargs: None,
            compute_preference_scores_fn=lambda **_kwargs: pl.DataFrame({"date": _dates(3), "bad": [1.0, 2.0, 3.0]}),
            compute_weights_from_target_profile_fn=lambda **_kwargs: pl.DataFrame({"date": _dates(3), "weight": [1 / 3] * 3}),
            assert_final_invariants_fn=lambda *_args, **_kwargs: None,
        )
