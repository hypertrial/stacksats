---
title: Recipe - Strict Validation
description: Task recipe for running strict validation and interpreting gates.
---

# Recipe: Strict Validation

## Goal

Run strict validation and diagnose failures quickly.

## Command

```bash
stacksats strategy validate \
  --strategy my_strategy.py:MyStrategy \
  --start-date 2020-01-01 \
  --end-date 2025-01-01 \
  --strict \
  --min-win-rate 50.0
```

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
