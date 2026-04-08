---
title: Animate Command
description: Reference for `stacksats strategy animate`.
---

# Animate Command

## Prerequisites

- A valid `backtest_result.json` artifact already exists.
- Output directory is writable.
- Install visual extras first: `pip install "stacksats[viz]"`.

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

## Expected output

- GIF file (default `strategy_vs_uniform_hd.gif`).
- Optional MP4/WebM video when `--video-format` is set.
- `animation_manifest.json` with `schema_version`, render metadata, source path, and strategy provenance.

## Key options

- `--window-mode rolling|non-overlapping` (default `rolling`). Use `non-overlapping` for shareable stakeholder-facing exports.
- `--max-frames <int>` to cap render size/time.
- `--fps`, `--width`, `--height` for output quality.
- `--video-format none|mp4|webm` to export a higher-quality video in addition to the GIF.

## Troubleshooting

- If JSON parse fails, verify `backtest_result.json` includes `schema_version`, `provenance`, `summary_metrics`, and `window_level_data`.
- If render is slow, lower `--fps`, `--max-frames`, or output dimensions.
- Video export requires a system `ffmpeg` binary. If you only need the compatibility artifact, leave `--video-format` at `none`.
- Prefer `mp4` for sharing. Use `webm` only when you specifically need it and your local `ffmpeg` build supports VP9.

## Next step

- Share alongside validation/backtest metrics, not as a standalone performance claim.
- Prefer the MP4 as the primary artifact and keep the GIF for compatibility.

## Feedback

- [Was this page helpful? Open docs feedback issue](https://github.com/hypertrial/stacksats/issues/new?template=docs_feedback.md&title=%5Bdocs%5D+Feedback%3A+Animate+Command)
