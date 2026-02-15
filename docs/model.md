---
title: Model Overview
description: High-level architecture of the StackSats dynamic Bitcoin DCA model.
---

# Model Overview

This page is the top-level map of the StackSats model.

The model computes daily DCA weights using MVRV and trend context while the framework enforces invariants (`sum == 1.0`, feasibility clipping, and lock semantics).

## Reading Path

1. [Signals and Features](concepts/model-signals-features.md)
2. [Allocation Kernel](concepts/allocation-kernel.md)
3. [Backtest Runtime](model_backtest.md)

## Model Architecture

```mermaid
flowchart LR
    subgraph Inputs
      P["Price history"]
      M["MVRV data"]
    end

    subgraph Features
      F1["Signals + lagging"]
      F2["Confidence and volatility"]
    end

    subgraph Intents
      I1["Raw daily intent"]
      I2["Dynamic multiplier"]
    end

    subgraph Framework
      K1["Feasibility clipping"]
      K2["Locked-prefix preservation"]
      K3["Remaining-budget enforcement"]
    end

    O["Final weights"]

    P --> Features
    M --> Features
    Features --> Intents
    Intents --> Framework
    Framework --> O
```

## Core properties

- Deterministic output for identical inputs.
- Historical allocations are immutable once locked.
- Future allocations are feasibility-aware and budget-constrained.

## Related references

- [Framework Boundary](framework.md)
- [Runtime Objects Overview](objects.md)
- [API Reference](reference/api/index.md)
