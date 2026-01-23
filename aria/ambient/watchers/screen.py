"""
Aria Ambient Intelligence - Screen Context Watcher

Monitors the current screen context to determine which world is active
and detect relevant activity patterns.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
import subprocess
import re

from .base import Watcher, WatcherConfig
from ..models import Signal
from ..constants import SignalType, CHECK_INTERVALS

logger = logging.getLogger(__name__)


class ScreenContextWatcher(Watcher):
    """
    Monitors screen context for world detection and activity tracking.

    This watcher provides:
    - Active application detection
    - Browser URL detection (Safari, Chrome)
    - User presence/focus detection
    - Context signals for world activation

    Privacy note: This watcher does NOT capture screen content,
    only application metadata.
    """

    name = "screen"
    description = "Monitors screen context for world detection"
    default_signal_type = SignalType.SCREEN_CONTEXT

    def __init__(self, config: WatcherConfig = None):
        config = config or WatcherConfig(
            check_interval=CHECK_INTERVALS.get("screen", 30),
        )
        super().__init__(config)

        self._last_app: Optional[str] = None
        self._last_url: Optional[str] = None
        self._app_durations: Dict[str, float] = {}  # Track time in apps

    async def observe(self) -> List[Signal]:
        """
        Observe current screen context.

        Returns signals when:
        - Application changes
        - URL changes (in browser)
        - Extended focus detected
        """
        signals = []

        try:
            current_app = self.get_active_app()
            current_url = self.get_active_url() if self._is_browser(current_app) else None

            # App change signal
            if current_app != self._last_app:
                if self._last_app:  # Don't signal on first run
                    signals.append(self.create_signal(
                        title=f"Switched to {current_app}",
                        content=f"User switched from {self._last_app} to {current_app}",
                        raw_data={
                            "event": "app_change",
                            "from_app": self._last_app,
                            "to_app": current_app,
                        }
                    ))
                self._last_app = current_app

            # URL change signal
            if current_url and current_url != self._last_url:
                domain = self._extract_domain(current_url)
                signals.append(self.create_signal(
                    title=f"Browsing: {domain}",
                    content=f"User navigated to {current_url}",
                    url=current_url,
                    raw_data={
                        "event": "url_change",
                        "url": current_url,
                        "domain": domain,
                        "browser": current_app,
                    }
                ))
                self._last_url = current_url

        except Exception as e:
            logger.error(f"Screen context observation error: {e}")

        return signals

    def get_active_app(self) -> str:
        """
        Get the name of the currently active application.

        Returns:
            Application name, or "Unknown" if detection fails
        """
        try:
            script = '''
            tell application "System Events"
                set frontApp to name of first application process whose frontmost is true
            end tell
            return frontApp
            '''
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.warning("Timeout getting active app")
        except Exception as e:
            logger.error(f"Error getting active app: {e}")

        return "Unknown"

    def get_active_url(self) -> Optional[str]:
        """
        Get the URL from the active browser tab.

        Supports Safari and Chrome.

        Returns:
            URL string, or None if not in a browser or detection fails
        """
        active_app = self._last_app or self.get_active_app()

        if "Safari" in active_app:
            return self._get_safari_url()
        elif "Chrome" in active_app or "Google Chrome" in active_app:
            return self._get_chrome_url()
        elif "Firefox" in active_app:
            return self._get_firefox_url()

        return None

    def _get_safari_url(self) -> Optional[str]:
        """Get URL from Safari."""
        try:
            script = '''
            tell application "Safari"
                if (count of windows) > 0 then
                    return URL of current tab of front window
                end if
            end tell
            return ""
            '''
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Error getting Safari URL: {e}")

        return None

    def _get_chrome_url(self) -> Optional[str]:
        """Get URL from Chrome."""
        try:
            script = '''
            tell application "Google Chrome"
                if (count of windows) > 0 then
                    return URL of active tab of front window
                end if
            end tell
            return ""
            '''
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Error getting Chrome URL: {e}")

        return None

    def _get_firefox_url(self) -> Optional[str]:
        """Get URL from Firefox (limited support)."""
        # Firefox doesn't have good AppleScript support
        # This is a placeholder for potential future implementation
        return None

    def is_user_present(self) -> bool:
        """
        Check if user is actively using the computer.

        Uses idle time to determine presence.

        Returns:
            True if user appears to be present
        """
        try:
            result = subprocess.run(
                ["ioreg", "-c", "IOHIDSystem"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                # Parse idle time from output
                match = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', result.stdout)
                if match:
                    idle_ns = int(match.group(1))
                    idle_seconds = idle_ns / 1_000_000_000
                    # Consider present if idle less than 5 minutes
                    return idle_seconds < 300
        except Exception as e:
            logger.debug(f"Error checking user presence: {e}")

        return True  # Assume present if check fails

    def is_user_focused(self) -> bool:
        """
        Check if user appears to be in a focused work state.

        Detects:
        - Do Not Disturb mode
        - Full-screen applications
        - Video call applications

        Returns:
            True if user appears to be in focus mode
        """
        active_app = self._last_app or self.get_active_app()

        # Check for video call apps
        focus_apps = ["zoom.us", "Microsoft Teams", "Slack", "FaceTime"]
        if any(app.lower() in active_app.lower() for app in focus_apps):
            return True

        # Check Do Not Disturb (macOS)
        try:
            result = subprocess.run(
                ["defaults", "-currentHost", "read", "com.apple.notificationcenterui", "doNotDisturb"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and result.stdout.strip() == "1":
                return True
        except Exception:
            pass

        return False

    def get_context(self) -> Dict[str, Any]:
        """
        Get complete current context.

        Returns:
            Dictionary with all context information
        """
        active_app = self.get_active_app()
        active_url = self.get_active_url() if self._is_browser(active_app) else None

        return {
            "active_app": active_app,
            "active_url": active_url,
            "domain": self._extract_domain(active_url) if active_url else None,
            "is_browser": self._is_browser(active_app),
            "user_present": self.is_user_present(),
            "user_focused": self.is_user_focused(),
        }

    def _is_browser(self, app_name: str) -> bool:
        """Check if app is a web browser."""
        browsers = ["safari", "chrome", "firefox", "brave", "edge", "opera", "arc"]
        return any(browser in app_name.lower() for browser in browsers)

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        if not url:
            return ""

        try:
            # Remove protocol
            if "://" in url:
                url = url.split("://", 1)[1]
            # Remove path
            domain = url.split("/")[0]
            # Remove port
            domain = domain.split(":")[0]
            return domain
        except Exception:
            return url
