"""
Lazy Anthropic SDK Loader

The anthropic SDK takes ~40 seconds to import due to 5000+ type modules.
This module provides lazy loading to avoid blocking startup.
"""

import os
from typing import Optional, TYPE_CHECKING

# Module-level cache
_anthropic = None
_client = None


def get_anthropic():
    """
    Get the anthropic module (lazy loaded).

    First call will take ~40s, subsequent calls are instant.
    """
    global _anthropic
    if _anthropic is None:
        print("Loading Anthropic SDK (first load takes ~30-40s)...", flush=True)
        import anthropic
        _anthropic = anthropic
        print("Anthropic SDK ready!", flush=True)
    return _anthropic


def get_client(api_key: Optional[str] = None):
    """
    Get a shared Anthropic client instance.

    Args:
        api_key: Optional API key. If not provided, uses ANTHROPIC_API_KEY env var.

    Returns:
        Anthropic client instance (shared singleton)
    """
    global _client
    if _client is None:
        anthropic = get_anthropic()
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        _client = anthropic.Anthropic(api_key=key)
    return _client


def is_loaded() -> bool:
    """Check if anthropic module is already loaded."""
    return _anthropic is not None
