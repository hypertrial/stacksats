---
title: Animate Command
description: Reference for `stacksats strategy animate`.
---

# Animate Command

## Prerequisites

- A valid `backtest_result.json` artifact already exists.
- Output directory is writable.

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

## Expected output

- GIF file (default `strategy_vs_uniform_hd.gif`).
- `animation_manifest.json` with render metadata and source path.

## Key options

- `--window-mode rolling|non-overlapping` (default `rolling`).
- `--max-frames <int>` to cap render size/time.
- `--fps`, `--width`, `--height` for output quality.

## Troubleshooting

- If JSON parse fails, verify artifact shape includes `window_level_data`.
- If render is slow, lower `--fps`, `--max-frames`, or output dimensions.

## Next step

- Share alongside validation/backtest metrics, not as a standalone performance claim.

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Animate+Command)
