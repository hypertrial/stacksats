---
title: Framework Boundary
description: Canonical contract separating user strategy scope from framework-enforced mechanics.
---

# Framework vs User Control

This document is the canonical strategy contract for StackSats.

## Framework Owns (Non-Negotiable)

1. Given a fixed budget: generalized to a proportion (total accumulation weight = 1).
2. Given a fixed allocation span (global config) between 90 and 1460 days (inclusive).
3. Must define a complete strategy:
   - Initializes all daily accumulation weights over the allocation span such that the sum of all weights equals the total budget (=1).
   - Day-by-day iterative updates to future weights, reshuffling the fixed budget.
   - Locked historical weights (past days are immutable).
4. Total budget must be used by the end of the allocation span (all daily weights must sum to 1).
5.
   - Minimum daily weight is `1e-5`.
   - Maximum daily weight is `0.1`.
6. Validation guards (`NaN`/`inf`/range checks) and final invariants.

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
