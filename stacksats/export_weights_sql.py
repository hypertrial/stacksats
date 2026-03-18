"""SQL payload helpers for export-weight database operations."""

from __future__ import annotations

from typing import Callable

import polars as pl


def prepare_copy_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Build temp-table COPY payload with canonical DB column names."""
    copy_df = df.select(
        ["day_index", "start_date", "end_date", "date", "price_usd", "weight"]
    )
    copy_df = copy_df.rename({
        "day_index": "id",
        "date": "DCA_date",
        "price_usd": "btc_usd",
    })
    copy_df = copy_df.with_columns(pl.col("id").cast(pl.Int64))
    copy_df = copy_df.with_columns(
        pl.when(pl.col("btc_usd").is_null() | ~pl.col("btc_usd").is_finite())
        .then(pl.lit(""))
        .otherwise(pl.col("btc_usd").cast(pl.Utf8))
        .alias("btc_usd"),
        pl.when(pl.col("weight").is_null() | ~pl.col("weight").is_finite())
        .then(pl.lit(""))
        .otherwise(pl.col("weight").cast(pl.Utf8))
        .alias("weight"),
    )
    return copy_df


def build_insert_rows(df: pl.DataFrame) -> list[tuple[int, object, object, object, float | None, float | None]]:
    """Build execute_values rows for INSERT fallback path."""
    normalized = df.select(
        pl.col("day_index").cast(pl.Int64).alias("day_index"),
        "start_date",
        "end_date",
        "date",
        pl.when(pl.col("price_usd").is_null() | ~pl.col("price_usd").is_finite())
        .then(None)
        .otherwise(pl.col("price_usd").cast(pl.Float64, strict=False))
        .alias("price_usd"),
        pl.when(pl.col("weight").is_null() | ~pl.col("weight").is_finite())
        .then(None)
        .otherwise(pl.col("weight").cast(pl.Float64, strict=False))
        .alias("weight"),
    )
    return list(
        zip(
            normalized["day_index"].to_list(),
            normalized["start_date"].to_list(),
            normalized["end_date"].to_list(),
            normalized["date"].to_list(),
            normalized["price_usd"].to_list(),
            normalized["weight"].to_list(),
            strict=True,
        )
    )


def build_update_rows(
    today_df: pl.DataFrame,
    *,
    current_btc_price: float | None,
) -> list[tuple]:
    """Build rows for bulk UPDATE statements."""
    normalized = today_df.select(
        pl.col("day_index").cast(pl.Int64).alias("day_index"),
        "start_date",
        "end_date",
        "date",
        pl.when(pl.col("weight").is_null() | ~pl.col("weight").is_finite())
        .then(None)
        .otherwise(pl.col("weight").cast(pl.Float64, strict=False))
        .alias("weight"),
    )
    base_rows = list(
        zip(
            normalized["day_index"].to_list(),
            normalized["start_date"].to_list(),
            normalized["end_date"].to_list(),
            normalized["date"].to_list(),
            normalized["weight"].to_list(),
            strict=True,
        )
    )
    if current_btc_price is None:
        return base_rows
    return [row + (current_btc_price,) for row in base_rows]


def build_values_sql(
    batch: list[tuple],
    *,
    include_price: bool,
    sql_quote: Callable[[object], str],
) -> str:
    """Render VALUES payload for a bulk UPDATE statement."""
    values_list: list[str] = []
    if include_price:
        for row in batch:
            (
                id_val,
                start_date_val,
                end_date_val,
                dca_date_val,
                weight_val,
                btc_usd_val,
            ) = row
            weight_sql = sql_quote(weight_val)
            btc_usd_sql = sql_quote(btc_usd_val)
            values_list.append(
                f"({id_val}, {sql_quote(start_date_val)}::date, "
                f"{sql_quote(end_date_val)}::date, "
                f"{sql_quote(dca_date_val)}::date, "
                f"{weight_sql}::float, {btc_usd_sql}::float)"
            )
    else:
        for row in batch:
            (id_val, start_date_val, end_date_val, dca_date_val, weight_val) = row
            weight_sql = sql_quote(weight_val)
            values_list.append(
                f"({id_val}, {sql_quote(start_date_val)}::date, "
                f"{sql_quote(end_date_val)}::date, "
                f"{sql_quote(dca_date_val)}::date, "
                f"{weight_sql}::float)"
            )
    return ", ".join(values_list)
