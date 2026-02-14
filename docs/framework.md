# Framework vs User Control

This document is the canonical strategy contract for StackSats.

## Framework Owns (Non-Negotiable)

1. Fixed budget.
2. Fixed allocation span (global config) between 90 and 1460 days (inclusive).
3. Uniform initialization of all daily weights.
4. Day-by-day iterative execution loop.
5. Daily reinitialization of all future days with the remaining uniform weight.
6. Locked historical weights (past days are immutable).
7. Feasibility projection/clipping at the daily handoff boundary.
8. Remaining-budget and allocation-range enforcement:
   - Total budget must be used by the end of the allocation span.
   - All daily weights must sum to 1.
   - Minimum daily weight is `1e-5`.
   - Maximum daily weight is `0.1`.
9. Validation guards (`NaN`/`inf`/range checks) and final invariants.

## User Owns (Flexible)

1. Feature engineering from lagged/base data
2. Signal definitions/formulas
3. Signal weights and hyperparameters
4. Daily intent output:
   - `propose_weight(state)` for per-day proposals, or
   - `build_target_profile(...)` for a full-window intent series

## Handoff Boundary

The user never writes the framework iteration loop.

User output (`proposed_weight_today` or daily profile intent) is handed to the framework allocation kernel, which computes `final_weight_today` by applying:

1. feasibility clipping
2. remaining-budget rules
3. historical lock rules
4. final invariant checks

## Required Behavior

1. Users can strongly influence allocation each day through features/signals/intent.
2. Users cannot alter iteration mechanics or rewrite past allocations.
3. Local, backtest, and production run the same sealed allocation kernel.

## Production Daily Lifecycle

1. Load locked historical weights for the active allocation span.
2. Build lagged features/signals using information available up to `current_date`.
3. Collect user daily intent (`proposed_weight_today` or profile-derived intent).
4. Project to feasible `final_weight_today` with remaining-budget constraints.
5. Persist today as locked.
6. Advance to next day; past values remain immutable.

## Design Intent

This boundary is deliberate: it maximizes strategy flexibility while preventing forward-looking bias and preserving deterministic allocation semantics.
