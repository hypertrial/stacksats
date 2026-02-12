# Strategy Validation Checklist (No Forward-Looking Bias)

Use this checklist before submitting or deploying a strategy.

It is designed to answer one core question:
**"Does my model avoid forward-looking leakage and behave robustly out-of-sample?"**

## Inputs You Need

- A strategy spec in format: `module_or_path:ClassName`
- Example: `examples/model_example.py:ExampleMVRVStrategy`

Set it once:

```bash
export STRATEGY_SPEC="examples/model_example.py:ExampleMVRVStrategy"
```

## 0) Environment and Install

From repo root:

```bash
pip install -e .
pip install -r requirements-dev.txt
```

## 1) Primary Submission Gate (Must Pass)

Run package-level validation:

```bash
stacksats-validate \
  --strategy "$STRATEGY_SPEC" \
  --start-date 2020-01-01 \
  --end-date 2025-01-01 \
  --min-win-rate 50.0
```

Pass criteria:

- Command exits successfully (`0`)
- Summary reports:
  - `Forward Leakage: True`
  - `Weight Constraints: True`
  - Win rate meets threshold

Fail criteria:

- Any leakage message such as `Forward leakage detected`
- Any weight constraint failure (negative weights or sum != 1.0)
- Win rate below `--min-win-rate`

## 2) Hard Forward-Leakage Regression Tests (Must Pass)

Run the targeted leakage and stability suites:

```bash
pytest tests/test_bdd_forward_looking.py -v
pytest tests/test_weight_stability.py -v
pytest tests/test_api_enhancements.py -k "forward_leakage or leaky" -v
```

What these cover:

- Forward-looking behavior scenarios (BDD)
- Past-weight stability as `current_date` advances
- Intentional leaky strategy detection in API validation

Pass criteria:

- All selected tests pass
- No forward-looking assertions fail

## 3) Time-Series Validation (Strongly Recommended)

Run walk-forward and out-of-sample consistency checks:

```bash
pytest tests/test_cross_validation.py -v
pytest tests/test_statistical_validation.py -v
```

What these cover:

- Expanding and rolling time-series folds (train before test)
- Overfitting signals (train/test gap checks)
- Randomized baseline checks (shuffled prices should not produce strong outperformance)

Pass criteria:

- No failures in cross-validation ordering or fold integrity
- No major overfitting/leakage indicator failures

## 4) Full Test Sweep Before Final Sign-Off

```bash
pytest tests/ -v
ruff check .
```

Pass criteria:

- Test suite passes or only has justified skips
- Lint passes

## 5) Optional Manual Sanity Backtest

Run a backtest and inspect output artifacts:

```bash
stacksats-backtest \
  --strategy "$STRATEGY_SPEC" \
  --start-date 2020-01-01 \
  --end-date 2025-01-01 \
  --output-dir output \
  --strategy-label pre-submit-check
```

Review:

- `output/metrics.json`
- Percentile behavior looks plausible (not suspiciously perfect)
- Win/loss profile is stable and explainable

## 6) Quick Go / No-Go Decision

Go if all are true:

- `stacksats-validate` passes with forward leakage and weight constraints marked true
- Forward-looking and stability tests pass
- Cross-validation/statistical validation show no leakage/overfitting red flags
- Full tests and lint pass

No-go if any are true:

- Any forward leakage failure
- Any weight normalization/negativity failure
- Reproducible evidence of suspicious performance on randomized/shuffled data
