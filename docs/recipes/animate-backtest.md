---
title: Recipe - Animate Backtest Performance
description: Generate a high-definition GIF showing strategy-vs-uniform performance over time.
---

# Recipe: Animate Backtest Performance

## Goal

Create a single HD GIF that communicates dynamic strategy performance versus uniform DCA over rolling windows.

## Command

```bash
stacksats strategy animate \
  --backtest-json output/<strategy_id>/<version>/<run_id>/backtest_result.json \
  --output-dir output/<strategy_id>/<version>/<run_id> \
  --output-name strategy_vs_uniform_hd.gif \
  --fps 20 \
  --width 1920 \
  --height 1080 \
  --max-frames 240 \
  --window-mode rolling
```

## What the animation shows

- Top panel: dynamic percentile versus uniform percentile over time.
- Bottom panel: total BTC bought advantage versus uniform DCA (`%`) to date, with positive/negative area fill.
- Overlay counters: current excess percentile, total BTC vs uniform (`%`), and win-rate-to-date.

## Window modes

- `rolling` (default):
  - Uses all eligible rolling windows, then deterministic downsampling (`--max-frames`).
  - Best for internal model diagnostics and timeline detail.
- `non-overlapping`:
  - Uses a non-overlapping subset of windows before downsampling.
  - Better for external communication where overlap bias can be misleading.

## Render-time tradeoffs

- Higher `--width`/`--height` increases clarity and file size.
- Higher `--fps` increases smoothness and size.
- `--max-frames` is the main runtime bound:
  - lower values render faster,
  - higher values preserve more timeline detail.

## Output artifacts

- GIF: `strategy_vs_uniform_hd.gif` (or `--output-name`)
- Manifest: `animation_manifest.json` with frame/render metadata

## Related docs

- [Animate Command](../run/animate.md)
- [Interpret Backtest Metrics](interpret-backtest.md)
