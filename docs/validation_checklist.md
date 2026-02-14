# Strategy Validation Properties (Exhaustive)

This is the canonical running list of properties that validation enforces for strategies.

Framework contract reference: `docs/framework.md`

## Validation Property Checklist

1. [ ] **Allocation span config is valid**
   - `STACKSATS_ALLOCATION_SPAN_DAYS` must be an integer in `[90, 1460]` (default `365`).

2. [ ] **Strategy cannot bypass framework orchestration**
   - Custom `compute_weights` overrides are rejected.

3. [ ] **Strategy must implement an allowed hook path**
   - At least one is required: `propose_weight(state)` or `build_target_profile(ctx, features_df, signals)`.

4. [ ] **`transform_features` output type is valid**
   - Must return a pandas `DataFrame`.

5. [ ] **`build_signals` output type is valid**
   - Must return `dict[str, pandas.Series]`.

6. [ ] **Signal and profile series shape/index contract holds**
   - Each series must be a pandas `Series`, with no duplicate index, ascending index, exact match to window index.

7. [ ] **Signal and profile series numeric validity holds**
   - Series values must be finite numeric (no `NaN`, `inf`, or non-numeric coercion failures).

8. [ ] **`propose_weight` outputs are finite**
   - Every proposal must be finite numeric.

9. [ ] **Target profile mode is valid**
   - Only `preference` and `absolute` modes are allowed.

10. [ ] **Allocation index monotonicity holds for temporal counting**
   - Allocation index used for `n_past` must be monotonic increasing.

11. [ ] **Locked-prefix structural validity holds**
   - `locked_weights` must be 1D, finite, values in `[0, 1]`, and length `<= n_past`.

12. [ ] **Locked-prefix budget feasibility holds**
   - Running sum of locked weights must never exceed total budget `1.0`.

13. [ ] **Locked-prefix daily bounds hold on contract-length windows**
   - When window length equals configured span, locked values must be within `[1e-5, 0.1]`.

14. [ ] **Per-day clipping obeys future feasibility constraints**
   - For contract-length windows, daily clipping enforces both day bounds and future-budget feasibility.
   - Infeasible bounds must hard-fail.

15. [ ] **Output weight vector structure is valid**
   - Final weights must be 1D and finite.

16. [ ] **Output weight values are valid**
   - No negative weights; no values above `1.0` (within tolerance).

17. [ ] **Output weights sum exactly to budget (within tolerance)**
   - Final weight sum must be `1.0`.

18. [ ] **Contract-length day bounds hold on final weights**
   - For span-length windows, each day weight must be within `[1e-5, 0.1]`.

19. [ ] **Daily bounds are globally feasible for span length**
   - Span must satisfy feasibility: `n_days * min_daily_weight <= 1.0 <= n_days * max_daily_weight`.

20. [ ] **Historical lock immutability is enforced in strict checks**
   - Injected locked prefix must be preserved exactly when recomputing under perturbed future features.

21. [ ] **Forward-leakage resistance: masked-future invariance**
   - Prefix weights at probe date must match when all future features are masked to `NaN`.

22. [ ] **Forward-leakage resistance: perturbed-future invariance**
   - Prefix weights at probe date must match when future features are strongly perturbed.

23. [ ] **Profile-only leakage resistance is enforced**
   - For profile-hook strategies (without propose hook), prefix profile values must be invariant under masked/perturbed future inputs.

24. [ ] **Strict mode forbids in-place feature mutation**
   - Strategy must not mutate `ctx.features_df` during weight computation.

25. [ ] **Strict mode forbids profile-build feature mutation**
   - Strategy must not mutate `ctx.features_df` during transform/signal/profile build path.

26. [ ] **Strict mode determinism holds**
   - Repeated runs with identical inputs must produce exactly matching weights (within `atol=1e-12`).

27. [ ] **Weight constraints hold across validation windows**
   - Validation windows must not violate sum, negativity, or (when applicable) min/max day bounds.

28. [ ] **Boundary saturation stays below strict threshold**
   - In strict mode, boundary-hit rate (days at `MIN` or `MAX`) must be `<= max_boundary_hit_rate` (default `0.85`).

29. [ ] **Cross-fold robustness minimum is met in strict mode (when fold checks run)**
   - Minimum fold win rate must be `>= min_fold_win_rate` (default `20.0`).
   - Fold checks are skipped when there is insufficient date range for at least two valid folds.

30. [ ] **Cross-fold instability is bounded in strict mode (when fold checks run)**
   - Fold win-rate standard deviation must be `<= max_fold_win_rate_std` (default `35.0`).
   - Fold checks are skipped when there is insufficient date range for at least two valid folds.

31. [ ] **Shuffled-null robustness threshold is met in strict mode (when shuffled checks run)**
   - Mean win rate on shuffled-price trials must be `<= max_shuffled_win_rate` (default `80.0`) across `shuffled_trials` (default `3`).
   - Shuffled checks are skipped when `PriceUSD_coinmetrics` is missing, the shuffled window is empty, or `shuffled_trials <= 0`.

32. [ ] **Global win-rate threshold is met**
   - Validation backtest win rate must be `>= min_win_rate` (default `50.0`).

33. [ ] **Validation date range must contain data**
   - Empty requested validation range is an automatic validation failure.

34. [ ] **Backtest path must generate windows**
   - Validation relies on backtest windows; if none are generated, validation cannot pass.
