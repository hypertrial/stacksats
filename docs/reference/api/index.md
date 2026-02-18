---
title: API Reference
description: Generated API documentation for core StackSats modules.
---

# API Reference

This section is generated from source code docstrings and signatures.

## Core modules

- [strategy_types](strategy-types.md)
- [runner](runner.md)
- [api module](api-module.md)

## Notes

- API reference documents callable surface area.
- Conceptual behavior and tradeoffs are documented in the [Concepts](../../model.md) section.
- Stability boundary:
  - Treat top-level `stacksats` exports plus documented modules in this section as public API.
  - Lower-level modules such as `stacksats.backtest`, `stacksats.prelude`, and `stacksats.export_weights` are implementation detail and may change between releases.
