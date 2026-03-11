---
title: Recipe - Strict Validation
description: Task recipe for running strict validation and interpreting gates.
---

# Recipe: Strict Validation

## Goal

Run strict validation and diagnose failures quickly.

## Command

Use the canonical validate command options as your source of truth:

- [Validate Command](../run/validate.md)

Typical strict run:

```bash
stacksats strategy validate \
  --strategy my_strategy.py:MyStrategy \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --strict \
  --min-win-rate 50.0
```

`--strict` is shown explicitly here for emphasis. `strategy validate` already enables strict validation by default.

## Expected output

A summary line similar to:

```text
Validation PASSED | Forward Leakage: True | Weight Constraints: True | Win Rate: 62.40% (>=50.00%: True)
```

## If it fails

1. Check leakage and determinism first.
2. Check weight sum/range constraints next.
3. Then review fold and shuffled-null robustness diagnostics.

## Related docs

- [Validation Checklist](../validation_checklist.md)
- [Framework Boundary](../framework.md)
