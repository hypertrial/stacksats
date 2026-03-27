---
title: API Reference
description: Generated API documentation for core StackSats modules.
---

# API Reference

Start with [Public API](../public-api.md) if you want the supported `1.x` imports, CLI boundary, and artifact contracts.

This section is generated from source code docstrings and signatures.
Treat it as internal reference material unless a symbol is re-exported from top-level `stacksats`.

## Core modules

- [eda](eda.md)
- [strategy_types](strategy-types.md)
- [runner](runner.md)
- [api module](api-module.md)

## Notes

- API reference documents callable surface area.
- Conceptual behavior and tradeoffs are documented in the [Concepts](../../model.md) section.
- Stability boundary:
  - Treat top-level `stacksats` exports, documented artifact payloads, and the documented CLI subset as the stable public API.
  - Generated module pages in this section are internal reference and may change between releases even when they remain documented.
  - See [Stability Policy](../../stability.md) for the canonical support and deprecation rules.
