"""Backward-compatible module for the experimental MVRVPlusStrategy."""

from .overlays.mvrv_plus import MVRVPlusStrategy, main

__all__ = ["MVRVPlusStrategy", "main"]


if __name__ == "__main__":  # pragma: no cover
    main()
