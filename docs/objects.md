# Fundamental Objects

This repository uses one current fundamental object and one proposed companion object:

- `strategy` (current): user-defined runtime intent, implemented as `BaseStrategy`.
- `StrategyTimeSeries` (proposed): typed wrapper for the final time-series dataframe.

The two objects separate responsibilities cleanly:

- `strategy` decides intent.
- `StrategyTimeSeries` carries normalized outputs plus schema-level meaning.

## 1) Object: `strategy` (Current)

In StackSats, a strategy is a class that subclasses `BaseStrategy` in `stacksats/strategy_types.py`.

### Role

A strategy owns user intent logic:

- feature transforms
- signal definitions
- either day-level intent (`propose_weight`) or full-window intent (`build_target_profile`)

A strategy does not own framework allocation mechanics:

- iterative allocation kernel
- clipping and budget feasibility
- locked-history semantics
- final invariants and contract enforcement

### Shape

`BaseStrategy` identity fields:

- `strategy_id: str` (default: `"custom-strategy"`)
- `version: str` (default: `"0.1.0"`)
- `description: str` (default: `""`)

Primary hooks:

- `transform_features(ctx) -> pd.DataFrame`
- `build_signals(ctx, features_df) -> dict[str, pd.Series]`
- `propose_weight(state) -> float` (optional intent path)
- `build_target_profile(ctx, features_df, signals) -> TargetProfile | pd.Series` (optional intent path)

Execution wrappers:

- `backtest(config=None, **kwargs)`
- `validate(config=None, **kwargs)`
- `export_weights(config=None, **kwargs)`

### Required Contract

A valid strategy must:

- subclass `BaseStrategy`
- implement at least one intent hook (`propose_weight` or `build_target_profile`)
- not override `compute_weights` (framework-owned boundary)

This is enforced by `validate_strategy_contract(...)`.

### Runtime Inputs

`StrategyContext` (immutable):

- `features_df`
- `start_date`
- `end_date`
- `current_date`
- `locked_weights` (optional)
- column selectors (`btc_price_col`, `mvrv_col`)

`DayState`:

- `current_date`
- `features`
- `remaining_budget`
- `day_index`
- `total_days`
- `uniform_weight`

`TargetProfile`:

- `values: pd.Series`
- `mode: "preference" | "absolute"`

### Lifecycle and Guarantees

1. `StrategyRunner` validates strategy contract.
2. Framework precomputes features.
3. Strategy hooks run to produce intent.
4. Framework converts intent into final weights via sealed kernel.
5. Framework validates output constraints.
6. Backtest/validation/export returns artifacts with provenance (`strategy_id`, `version`, `config_hash`, `run_id`).

Output invariants:

- finite numeric series
- exact window-index alignment
- sum to `1.0` (tolerance)
- non-negative values
- within `[MIN_DAILY_WEIGHT, MAX_DAILY_WEIGHT]` for full-span windows

## 2) Object: `StrategyTimeSeries` (Proposed)

`StrategyTimeSeries` is the recommended second fundamental object. It wraps the final dataframe in a typed, validated, versioned container.

### Why this object exists

- makes final output explicit and stable
- prevents ad hoc dataframe assumptions
- centralizes schema definitions and column semantics
- creates a consistent boundary for file export and downstream consumers

### Name

Canonical name: `StrategyTimeSeries`.

Rationale:

- precise domain meaning
- direct mapping to final daily output
- extensible for future strategy output columns

### Contents

`StrategyTimeSeries` should contain metadata plus data payload.

Metadata (required):

- `strategy_id: str`
- `strategy_version: str`
- `run_id: str`
- `config_hash: str`
- `schema_version: str`
- `generated_at: pd.Timestamp`

Data payload (required `pd.DataFrame`):

- canonical daily date axis (`date` column or datetime index), unique and sorted
- `weight` column (`float`, finite, non-negative)

Useful optional columns:

- `locked: bool`
- `current_date: datetime`
- any strategy-produced columns that pass schema registration

Optional context:

- `window_start`
- `window_end`
- `source_columns` mapping
- `tags` or `notes`

Derived quality stats:

- row count
- date range
- weight sum
- invariant pass/fail flags

### Handwritten Schema Methods

This object should expose explicit schema APIs (handwritten, not inferred):

- `schema()`
- `schema_markdown()`
- `validate_schema_coverage()`

`schema()` returns per-column specs (authoritative and versioned).  
`schema_markdown()` renders a readable table for docs/UI.  
`validate_schema_coverage()` fails if dataframe columns exist without handwritten specs.

### Column Spec Model

Suggested `ColumnSpec` fields:

- `name`
- `dtype`
- `required`
- `description` (handwritten meaning)
- `unit` (optional)
- `constraints` (range/nullability/enum)
- `source` (framework/strategy/derived)
- `formula` (optional)

Example handwritten meanings:

- `date`: calendar day for allocation row; daily grain
- `weight`: final feasible daily allocation after framework clipping, lock, and budget rules
- `locked`: historical immutability marker; `True` means value is fixed

### Validation Rules

At construction or explicit validation, enforce:

- required columns present
- no duplicate dates
- sorted ascending dates
- finite numeric values where expected
- schema coverage for all exposed columns
- weight invariants where applicable

### Minimal Interface Sketch

```python
@dataclass(frozen=True)
class StrategyTimeSeries:
    metadata: StrategySeriesMetadata
    data: pd.DataFrame

    def schema(self) -> dict[str, ColumnSpec]: ...
    def schema_markdown(self) -> str: ...
    def validate_schema_coverage(self) -> None: ...
    def validate(self) -> None: ...
    def to_dataframe(self) -> pd.DataFrame: ...
```

## Integration Direction (No Code Changes in This Doc)

Recommended adoption path:

1. Introduce `StrategyTimeSeries` as a non-breaking companion return type.
2. Start with export boundary (`StrategyRunner.export`) where final tabular output is already explicit.
3. Keep dataframe compatibility while downstream code migrates.
4. Promote `StrategyTimeSeries` to canonical output object once stable.

## References

- `stacksats/strategy_types.py`
- `stacksats/runner.py`
- `docs/framework.md`
- `docs/model.md`
- `README.md`
