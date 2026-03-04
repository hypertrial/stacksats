from __future__ import annotations

import pandas as pd
import pytest

from stacksats.feature_materialization import (
    build_observed_frame,
    hash_dataframe,
    materialize_versioned_observations,
)


def test_build_observed_frame_restricts_to_observed_prefix() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    frame = pd.DataFrame({"x": range(5)}, index=idx)

    observed = build_observed_frame(
        frame,
        start_date="2024-01-02",
        current_date="2024-01-04",
    )

    assert list(observed.index) == list(idx[1:4])


def test_build_observed_frame_handles_empty_frame() -> None:
    observed = build_observed_frame(
        pd.DataFrame(),
        start_date="2024-01-01",
        current_date="2024-01-03",
    )

    assert list(observed.index) == list(pd.date_range("2024-01-01", "2024-01-03", freq="D"))


def test_materialize_versioned_observations_selects_latest_available_revision() -> None:
    observations = pd.DataFrame(
        {
            "effective_date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "available_at": ["2024-01-01", "2024-01-03", "2024-01-02"],
            "revision_id": [1, 2, 1],
            "signal": [1.0, 2.0, 3.0],
        }
    )

    as_of_old = materialize_versioned_observations(observations, as_of_date="2024-01-02")
    as_of_new = materialize_versioned_observations(observations, as_of_date="2024-01-03")

    assert float(as_of_old.loc[pd.Timestamp("2024-01-01"), "signal"]) == 1.0
    assert float(as_of_new.loc[pd.Timestamp("2024-01-01"), "signal"]) == 2.0


def test_materialize_versioned_observations_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        materialize_versioned_observations(
            pd.DataFrame({"effective_date": ["2024-01-01"]}),
            as_of_date="2024-01-01",
        )


def test_hash_dataframe_is_stable_for_equal_content() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    frame_a = pd.DataFrame({"x": [1.0, None]}, index=idx)
    frame_b = pd.DataFrame({"x": [1.0, None]}, index=idx)

    assert hash_dataframe(frame_a) == hash_dataframe(frame_b)

