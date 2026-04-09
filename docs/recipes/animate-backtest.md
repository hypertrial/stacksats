---
title: Recipe - Animate Backtest Performance
description: Generate a high-definition GIF showing strategy-vs-uniform performance over time.
---

# Recipe: Animate Backtest Performance

## Goal

Create a single HD animation that communicates cumulative strategy outcome versus uniform DCA while retaining per-window percentile context.

## Command

```bash
stacksats strategy animate \
  --backtest-json output/<strategy_id>/<version>/<run_id>/backtest_result.json \
  --output-dir output/<strategy_id>/<version>/<run_id> \
  --output-name strategy_vs_uniform_hd.gif \
  --video-format mp4 \
  --fps 20 \
  --width 1920 \
  --height 1080 \
  --max-frames 240 \
  --window-mode non-overlapping
```

## What the animation shows

- Top panel: cumulative BTC advantage versus uniform DCA (`%`) to date, with positive/negative area fill.
- Bottom panel: dynamic percentile versus uniform percentile for each selected window.
- Overlay counters: current excess percentile and win-rate-to-date, shown as annotations on the cumulative panel.

## Window modes

- `rolling` (default):
  - Uses all eligible rolling windows, then deterministic downsampling (`--max-frames`).
  - Best for internal model diagnostics and timeline detail.
- `non-overlapping`:
  - Uses a non-overlapping subset of windows before downsampling.
  - Recommended for external communication where overlap bias can be misleading.

## Render-time tradeoffs

- Higher `--width`/`--height` increases clarity and file size.
- Higher `--fps` increases smoothness and size.
- `--max-frames` is the main runtime bound:
  - lower values render faster,
  - higher values preserve more timeline detail.

## Output artifacts

- GIF: `strategy_vs_uniform_hd.gif` (or `--output-name`)
- Video: `strategy_vs_uniform_hd.mp4` or `.webm` when `--video-format` is set
- Manifest: `animation_manifest.json` with frame/render metadata

## Shareable defaults

- Use `--window-mode non-overlapping` for stakeholder-facing exports.
- Use `--video-format mp4` when you want a sharper primary artifact than GIF.
- Keep the GIF for compatibility and lightweight previews.
- Video export requires a system `ffmpeg` binary.

## Related docs

- [Animate Command](../run/animate.md)
- [Interpret Backtest Metrics](interpret-backtest.md)
