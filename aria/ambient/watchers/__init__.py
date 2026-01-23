"""
Aria Ambient Intelligence - Watchers

Watchers are continuous signal collectors that monitor various data sources
and produce signals for the cortex to process.

Available Watchers:
- Watcher: Abstract base class for all watchers
- WatcherConfig: Configuration for watchers
- WatcherScheduler: Background scheduler for running watchers
- NewsWatcher: Monitors news sources for relevant articles
- CalendarWatcher: Monitors calendar for events and reminders
- ScreenWatcher: Monitors screen context for relevant activity
- SocialWatcher: Monitors social media for mentions and trends
"""

from .base import Watcher, WatcherConfig
from .scheduler import WatcherScheduler
from .news import NewsWatcher
from .calendar import CalendarWatcher
from .screen import ScreenContextWatcher

# Social watcher will be added when implemented
# from .social import SocialWatcher

__all__ = [
    "Watcher",
    "WatcherConfig",
    "WatcherScheduler",
    "NewsWatcher",
    "CalendarWatcher",
    "ScreenContextWatcher",
    # "SocialWatcher",
]
