"""Backward-compatible module for the experimental ExampleMVRVStrategy."""

from .overlays.example_mvrv import ExampleMVRVStrategy, main

__all__ = ["ExampleMVRVStrategy", "main"]


if __name__ == "__main__":  # pragma: no cover
    main()
