"""
Aria Notifications - Push Notification Delivery

Uses ntfy.sh for cross-platform push notifications:
- Mobile: iOS/Android ntfy apps
- Desktop: Browser notifications
- Self-hostable for privacy
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from .config import DATA_PATH, NTFY_SERVER, NTFY_TOPIC, NTFY_TOKEN

logger = logging.getLogger(__name__)

# Configuration
NTFY_CONFIG_FILE = DATA_PATH / "ntfy_config.json"


class Priority(Enum):
    MIN = "min"
    LOW = "low"
    DEFAULT = "default"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class NotificationConfig:
    """ntfy.sh configuration."""
    server_url: str = "https://ntfy.sh"
    topic: str = "aria-notifications"  # User should change this!
    auth_token: Optional[str] = None
    default_priority: Priority = Priority.DEFAULT

    def to_dict(self) -> dict:
        return {
            "server_url": self.server_url,
            "topic": self.topic,
            "auth_token": self.auth_token,
            "default_priority": self.default_priority.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NotificationConfig":
        return cls(
            server_url=data.get("server_url", "https://ntfy.sh"),
            topic=data.get("topic", "aria-notifications"),
            auth_token=data.get("auth_token"),
            default_priority=Priority(data.get("default_priority", "default")),
        )

    @classmethod
    def from_env(cls) -> "NotificationConfig":
        """Load from environment variables."""
        return cls(
            server_url=NTFY_SERVER or "https://ntfy.sh",
            topic=NTFY_TOPIC or "aria-notifications",
            auth_token=NTFY_TOKEN or None,
        )


class Notifier:
    """
    Push notification sender using ntfy.sh.

    Usage:
        notifier = Notifier()
        await notifier.send("Title", "Body", priority="high")
    """

    def __init__(self, config: NotificationConfig = None):
        self.config = config or self._load_config()
        self._client: Optional[httpx.AsyncClient] = None
        self._sent_count = 0
        self._last_sent: Optional[str] = None

    def _load_config(self) -> NotificationConfig:
        """Load config from file or environment."""
        if NTFY_CONFIG_FILE.exists():
            try:
                with open(NTFY_CONFIG_FILE) as f:
                    data = json.load(f)
                    return NotificationConfig.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load ntfy config: {e}")

        return NotificationConfig.from_env()

    def save_config(self):
        """Save current config to file."""
        try:
            with open(NTFY_CONFIG_FILE, "w") as f:
                json.dump(self.config.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save ntfy config: {e}")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def send(
        self,
        title: str,
        body: str,
        priority: str = None,
        tags: List[str] = None,
        click_url: str = None,
        actions: List[Dict[str, str]] = None,
        attachment_url: str = None,
    ) -> bool:
        """
        Send a push notification.

        Args:
            title: Notification title
            body: Notification body text
            priority: Priority level (min, low, default, high, urgent)
            tags: List of tags (can include emoji shortcodes)
            click_url: URL to open when notification is clicked
            actions: List of action buttons
            attachment_url: URL to attach (image, file)

        Returns:
            True if sent successfully
        """
        # Check if configured
        if self.config.topic == "aria-notifications":
            logger.warning("ntfy topic not configured. Using default topic (not recommended).")

        client = await self._get_client()

        url = f"{self.config.server_url}/{self.config.topic}"

        headers = {
            "Title": title,
            "Priority": priority or self.config.default_priority.value,
        }

        if tags:
            headers["Tags"] = ",".join(tags)

        if click_url:
            headers["Click"] = click_url

        if actions:
            headers["Actions"] = json.dumps(actions)

        if attachment_url:
            headers["Attach"] = attachment_url

        if self.config.auth_token:
            headers["Authorization"] = f"Bearer {self.config.auth_token}"

        try:
            response = await client.post(url, content=body, headers=headers)

            if response.status_code == 200:
                self._sent_count += 1
                self._last_sent = datetime.now().isoformat()
                logger.info(f"Notification sent: {title}")
                return True
            else:
                logger.error(f"Notification failed: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Notification error: {e}")
            return False

    async def send_briefing(self, briefing: str, priority: str = "default") -> bool:
        """Send a briefing notification."""
        # Split long briefings
        max_length = 4000  # ntfy limit

        if len(briefing) <= max_length:
            return await self.send(
                title="Aria Briefing",
                body=briefing,
                priority=priority,
                tags=["clipboard", "brain"],  # Emoji tags
            )

        # Split into parts
        parts = [briefing[i:i+max_length] for i in range(0, len(briefing), max_length)]

        for i, part in enumerate(parts):
            await self.send(
                title=f"Aria Briefing ({i+1}/{len(parts)})",
                body=part,
                priority=priority,
                tags=["clipboard", "brain"],
            )

        return True

    async def send_insight(self, title: str, body: str, priority: str = "high") -> bool:
        """Send an insight notification."""
        return await self.send(
            title=f"Aria Insight: {title}",
            body=body,
            priority=priority,
            tags=["bulb", "bell"],
        )

    async def send_reminder(self, title: str, body: str) -> bool:
        """Send a reminder notification."""
        return await self.send(
            title=title,
            body=body,
            priority="high",
            tags=["alarm_clock"],
        )

    async def send_alert(self, title: str, body: str) -> bool:
        """Send an urgent alert."""
        return await self.send(
            title=f"Warning: {title}",
            body=body,
            priority="urgent",
            tags=["warning", "rotating_light"],
        )

    def configure(
        self,
        server_url: str = None,
        topic: str = None,
        auth_token: str = None,
    ):
        """Update notification configuration."""
        if server_url:
            self.config.server_url = server_url
        if topic:
            self.config.topic = topic
        if auth_token is not None:
            self.config.auth_token = auth_token

        self.save_config()
        logger.info("Notification config updated")

    def get_status(self) -> dict:
        """Get notifier status."""
        return {
            "configured": bool(self.config.topic != "aria-notifications"),
            "server_url": self.config.server_url,
            "topic": self.config.topic,
            "sent_count": self._sent_count,
            "last_sent": self._last_sent,
        }

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Singleton
_notifier: Optional[Notifier] = None


def get_notifier() -> Notifier:
    """Get singleton notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = Notifier()
    return _notifier


# ============================================================================
# DELIVERY ENGINE INTEGRATION
# ============================================================================

async def notification_delivery_handler(action) -> bool:
    """
    Handler for DeliveryEngine to send notifications.

    This integrates with aria/ambient/delivery/engine.py
    """
    notifier = get_notifier()

    # Extract info from PreparedAction
    title = action.message[:50] if action.message else "Aria Update"
    body = action.message or ""

    # Determine priority from action
    priority_map = {
        "critical": "urgent",
        "high": "high",
        "medium": "default",
        "low": "low",
    }

    # Handle both enum and string priority values
    priority_value = action.priority
    if hasattr(priority_value, 'value'):
        priority_value = priority_value.value
    priority = priority_map.get(str(priority_value).lower(), "default")

    return await notifier.send(title=title, body=body, priority=priority)


# ============================================================================
# SYNC WRAPPER FOR MCP TOOLS
# ============================================================================

def send_notification_sync(title: str, body: str, priority: str = "default") -> bool:
    """Synchronous wrapper for sending notifications."""
    notifier = get_notifier()

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create new event loop in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, notifier.send(title, body, priority))
                return future.result(timeout=10)
        else:
            return asyncio.run(notifier.send(title, body, priority))
    except Exception as e:
        logger.error(f"Sync notification error: {e}")
        return False
