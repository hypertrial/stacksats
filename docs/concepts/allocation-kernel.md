---
title: Allocation Kernel
description: Framework-enforced allocation mechanics applied to strategy intent.
---

# Allocation Kernel

The allocation kernel is framework-owned. Strategies provide intent; the framework enforces feasibility and invariants.

## Kernel responsibilities

- Preserve immutable locked prefix.
- Clip daily intent at each handoff to satisfy future feasibility.
- Enforce remaining-budget constraints.
- Enforce final invariants (finite, non-negative, sum to `1.0`).

## Contract boundary

```mermaid
flowchart LR
    U["User strategy intent"] --> K1["Clip to feasible daily bounds"]
    K1 --> K2["Preserve locked history"]
    K2 --> K3["Distribute remaining budget"]
    K3 --> O["Final weights"]
```

## Core behavior

```python
def allocate_from_proposals(proposals, n_past, n_total, locked_weights=None):
    # 1) Preserve locked prefix exactly
    # 2) Clip unlocked past/current days
    # 3) Distribute future remainder uniformly
    # 4) Validate invariants
```

## Constants

| Constant | Value |
| --- | --- |
| `MIN_DAILY_WEIGHT` | `1e-5` |
| `MAX_DAILY_WEIGHT` | `0.1` |
| Typical allocation span | `365` days (configurable) |

## Enforcement modules

- `stacksats/framework_contract.py`
- `stacksats/model_development.py`
- `stacksats/prelude.py`

## Related docs

- [Framework Boundary](../framework.md)
- [Validation Checklist](../validation_checklist.md)
