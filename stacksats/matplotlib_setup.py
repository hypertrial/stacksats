"""Matplotlib runtime environment setup for constrained environments."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _is_writable_dir(path: Path) -> bool:
    """Return True when path exists (or can be created) and is writable."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def configure_matplotlib_env() -> None:
    """Ensure Matplotlib/font cache points to a writable location.

    Some execution environments (CI, sandboxes, remote runners) have non-writable
    HOME cache locations, causing repeated Matplotlib and fontconfig warnings.
    This function redirects cache/config to a writable `.stacksats` directory.
    """
    cache_root = Path.home() / ".stacksats"
    xdg_cache = cache_root / "cache"
    mpl_config = cache_root / "matplotlib"

    if not _is_writable_dir(xdg_cache) or not _is_writable_dir(mpl_config):
        # Last-resort fallback if HOME is not writable.
        # Use a stable temp location so font cache can be reused across runs.
        tmp_root = Path(tempfile.gettempdir()) / "stacksats-cache"
        xdg_cache = tmp_root / "cache"
        mpl_config = tmp_root / "matplotlib"
        xdg_cache.mkdir(parents=True, exist_ok=True)
        mpl_config.mkdir(parents=True, exist_ok=True)

    # Override when missing OR unusable to suppress noisy runtime warnings.
    current_xdg = os.environ.get("XDG_CACHE_HOME")
    if not current_xdg or not _is_writable_dir(Path(current_xdg)):
        os.environ["XDG_CACHE_HOME"] = str(xdg_cache)

    current_mpl = os.environ.get("MPLCONFIGDIR")
    if not current_mpl or not _is_writable_dir(Path(current_mpl)):
        os.environ["MPLCONFIGDIR"] = str(mpl_config)


# Configure eagerly on import so callers can simply import this module
# before importing matplotlib without introducing E402 lint issues.
configure_matplotlib_env()
