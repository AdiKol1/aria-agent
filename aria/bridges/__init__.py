"""
Aria Bridges - Multi-Platform Messaging Access

Bridges allow accessing Aria from various messaging platforms.
"""

from typing import Optional

_telegram_bridge: Optional["TelegramBridge"] = None


def get_telegram_bridge():
    """Get Telegram bridge singleton."""
    global _telegram_bridge
    if _telegram_bridge is None:
        from .telegram import TelegramBridge
        _telegram_bridge = TelegramBridge()
    return _telegram_bridge


__all__ = ["get_telegram_bridge"]
