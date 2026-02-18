"""SQL payload helpers for export-weight database operations."""

from __future__ import annotations

from typing import Callable

import pandas as pd


def prepare_copy_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Build temp-table COPY payload with canonical DB column names."""
    copy_df = df[
        ["day_index", "start_date", "end_date", "date", "price_usd", "weight"]
    ].copy()
    copy_df = copy_df.rename(
        columns={
            "day_index": "id",
            "date": "DCA_date",
            "price_usd": "btc_usd",
        }
    )
    copy_df["id"] = copy_df["id"].astype(int)
    copy_df["btc_usd"] = copy_df["btc_usd"].where(pd.notna(copy_df["btc_usd"]), "")
    copy_df["weight"] = copy_df["weight"].where(pd.notna(copy_df["weight"]), "")
    return copy_df


def build_insert_rows(df: pd.DataFrame) -> list[tuple[int, object, object, object, float | None, float | None]]:
    """Build execute_values rows for INSERT fallback path."""
    return [
        (
            int(row["day_index"]),
            row["start_date"],
            row["end_date"],
            row["date"],
            float(row["price_usd"]) if pd.notna(row["price_usd"]) else None,
            float(row["weight"]) if pd.notna(row["weight"]) else None,
        )
        for _, row in df.iterrows()
    ]


def build_update_rows(
    today_df: pd.DataFrame,
    *,
    current_btc_price: float | None,
) -> list[tuple]:
    """Build rows for bulk UPDATE statements."""
    if current_btc_price is not None:
        return [
            (
                int(row["day_index"]),
                row["start_date"],
                row["end_date"],
                row["date"],
                float(row["weight"]) if pd.notna(row["weight"]) else None,
                current_btc_price,
            )
            for _, row in today_df.iterrows()
        ]
    return [
        (
            int(row["day_index"]),
            row["start_date"],
            row["end_date"],
            row["date"],
            float(row["weight"]) if pd.notna(row["weight"]) else None,
        )
        for _, row in today_df.iterrows()
    ]


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
