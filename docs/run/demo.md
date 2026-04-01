---
title: Demo Command
description: Reference for the packaged offline demo commands.
---

# Demo Command

Use the demo commands for the fastest first successful run.

## Most common command

```bash
stacksats demo backtest
```

This uses:

- packaged demo parquet bundled with the wheel
- default strategy `simple-zscore`
- default date bounds `2018-01-01` to `2025-12-31`

## Commands

```bash
stacksats demo validate
stacksats demo backtest
stacksats demo export
```

## Notes

- No Google Drive setup is required.
- No `STACKSATS_ANALYTICS_PARQUET` environment variable is required.
- Artifacts are written under `output/<strategy_id>/<version>/<run_id>/`.

## Next Steps

- [Quickstart](../start/quickstart.md)
- [Full Data Setup](../start/full-data-setup.md)
