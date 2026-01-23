"""
Aria Ambient Intelligence - Calendar Watcher

Monitors calendar for upcoming events, deadlines, and scheduling opportunities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import subprocess
import json

from .base import Watcher, WatcherConfig
from ..models import Signal
from ..constants import SignalType, CHECK_INTERVALS

logger = logging.getLogger(__name__)


class CalendarWatcher(Watcher):
    """
    Monitors calendar for events and generates relevant signals.

    Features:
    - Upcoming event detection
    - Deadline reminders
    - Meeting preparation alerts
    - Free time detection

    Currently supports:
    - Apple Calendar (macOS native)

    Future support:
    - Google Calendar API
    - Microsoft Outlook/365
    """

    name = "calendar"
    description = "Monitors calendar for events and deadlines"
    default_signal_type = SignalType.CALENDAR_EVENT

    def __init__(self, config: WatcherConfig = None):
        config = config or WatcherConfig(
            check_interval=CHECK_INTERVALS.get("calendar", 600),
            custom_settings={
                "lookahead_days": 7,
                "reminder_minutes": [60, 15],  # When to send reminders
            }
        )
        super().__init__(config)

        self._known_events: Dict[str, datetime] = {}  # event_id -> last_notified
        self._last_check: Optional[datetime] = None

    async def observe(self) -> List[Signal]:
        """
        Check calendar and generate signals for relevant events.

        Generates signals for:
        - Events starting soon
        - Events starting in the next hour
        - Deadlines approaching
        """
        signals = []
        now = datetime.now()

        try:
            # Get upcoming events
            lookahead = self.get_setting("lookahead_days", 7)
            events = self._get_upcoming_events(days=lookahead)

            for event in events:
                event_signals = self._process_event(event, now)
                signals.extend(event_signals)

            self._last_check = now

        except Exception as e:
            logger.error(f"Calendar observation error: {e}")

        return signals

    def _process_event(
        self,
        event: Dict[str, Any],
        now: datetime
    ) -> List[Signal]:
        """Process a single event and generate relevant signals."""
        signals = []

        event_id = event.get("id", str(hash(event.get("title", ""))))
        start_time = event.get("start_time")

        if not start_time:
            return signals

        # Parse start time if string
        if isinstance(start_time, str):
            try:
                start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                start_time = start_time.replace(tzinfo=None)  # Make naive for comparison
            except ValueError:
                return signals

        time_until = start_time - now
        minutes_until = time_until.total_seconds() / 60

        # Check reminder thresholds
        reminder_minutes = self.get_setting("reminder_minutes", [60, 15])

        for reminder_threshold in reminder_minutes:
            # Within threshold window (reminder_threshold to reminder_threshold-check_interval)
            threshold_start = reminder_threshold
            threshold_end = max(0, reminder_threshold - self._config.check_interval / 60)

            if threshold_end <= minutes_until <= threshold_start:
                # Check if we already sent this reminder
                reminder_key = f"{event_id}_{reminder_threshold}"
                if reminder_key not in self._known_events:
                    self._known_events[reminder_key] = now

                    if minutes_until <= 15:
                        signal_type = SignalType.CALENDAR_REMINDER
                        title = f"Starting soon: {event.get('title', 'Event')}"
                        priority_hint = "high"
                    else:
                        signal_type = SignalType.CALENDAR_EVENT
                        title = f"Upcoming: {event.get('title', 'Event')}"
                        priority_hint = "medium"

                    signals.append(self.create_signal(
                        title=title,
                        content=self._format_event_content(event, minutes_until),
                        signal_type=signal_type,
                        raw_data={
                            "event": event,
                            "minutes_until": minutes_until,
                            "priority_hint": priority_hint,
                        }
                    ))
                    break  # Only one reminder per check

        return signals

    def _format_event_content(
        self,
        event: Dict[str, Any],
        minutes_until: float
    ) -> str:
        """Format event details into readable content."""
        parts = []

        title = event.get("title", "Untitled")
        parts.append(f"Event: {title}")

        if minutes_until < 60:
            parts.append(f"Starts in {int(minutes_until)} minutes")
        else:
            hours = int(minutes_until / 60)
            parts.append(f"Starts in {hours} hour(s)")

        if event.get("location"):
            parts.append(f"Location: {event['location']}")

        if event.get("notes"):
            notes = event["notes"][:200] + "..." if len(event.get("notes", "")) > 200 else event["notes"]
            parts.append(f"Notes: {notes}")

        if event.get("attendees"):
            attendees = event["attendees"][:3]
            parts.append(f"With: {', '.join(attendees)}")
            if len(event["attendees"]) > 3:
                parts.append(f"  and {len(event['attendees']) - 3} others")

        return "\n".join(parts)

    def _get_upcoming_events(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get upcoming calendar events from Apple Calendar.

        Returns:
            List of event dictionaries
        """
        events = []

        try:
            # Calculate date range
            now = datetime.now()
            end_date = now + timedelta(days=days)

            # AppleScript to get events
            script = f'''
            set now_date to current date
            set end_date to now_date + ({days} * days)

            set event_list to {{}}

            tell application "Calendar"
                set all_calendars to calendars
                repeat with cal in all_calendars
                    try
                        set cal_events to (every event of cal whose start date >= now_date and start date <= end_date)
                        repeat with e in cal_events
                            set event_info to {{}}
                            set end of event_info to ("title:" & (summary of e as text))
                            set end of event_info to ("start:" & (start date of e as text))
                            set end of event_info to ("end:" & (end date of e as text))
                            try
                                set end of event_info to ("location:" & (location of e as text))
                            end try
                            try
                                set end of event_info to ("notes:" & (description of e as text))
                            end try
                            set end of event_info to ("calendar:" & (name of cal as text))
                            set end of event_list to event_info
                        end repeat
                    end try
                end repeat
            end tell

            return event_list
            '''

            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0 and result.stdout.strip():
                events = self._parse_applescript_events(result.stdout)

        except subprocess.TimeoutExpired:
            logger.warning("Timeout getting calendar events")
        except Exception as e:
            logger.error(f"Error getting calendar events: {e}")

        return events

    def _parse_applescript_events(self, output: str) -> List[Dict[str, Any]]:
        """Parse AppleScript output into event dictionaries."""
        events = []

        try:
            # AppleScript returns nested lists, parse them
            # Format: {{title:..., start:..., ...}, {...}}

            # Simple parsing - split by event boundaries
            output = output.strip()
            if not output or output == "{}":
                return events

            # Extract event blocks
            current_event = {}
            for part in output.split(", "):
                part = part.strip().strip("{}")
                if ":" in part:
                    key, _, value = part.partition(":")
                    key = key.strip().lower()
                    value = value.strip()

                    if key == "title" and current_event:
                        # New event starting
                        if current_event.get("title"):
                            events.append(current_event)
                        current_event = {}

                    if key == "start":
                        # Parse date
                        try:
                            # macOS date format varies, try common formats
                            for fmt in ["%A, %B %d, %Y at %I:%M:%S %p",
                                       "%B %d, %Y at %I:%M:%S %p",
                                       "%Y-%m-%d %H:%M:%S"]:
                                try:
                                    dt = datetime.strptime(value, fmt)
                                    current_event["start_time"] = dt.isoformat()
                                    break
                                except ValueError:
                                    continue
                        except Exception:
                            current_event["start_time_raw"] = value
                    else:
                        current_event[key] = value

            # Add last event
            if current_event.get("title"):
                events.append(current_event)

        except Exception as e:
            logger.error(f"Error parsing calendar events: {e}")

        return events

    def get_upcoming(self, days: int = 7) -> List[Signal]:
        """
        Get signals for all upcoming events.

        This is a synchronous helper for querying events without
        going through the normal observation cycle.

        Args:
            days: Number of days to look ahead

        Returns:
            List of signals for upcoming events
        """
        signals = []
        events = self._get_upcoming_events(days)
        now = datetime.now()

        for event in events:
            start_time = event.get("start_time")
            if not start_time:
                continue

            if isinstance(start_time, str):
                try:
                    start_time = datetime.fromisoformat(start_time)
                except ValueError:
                    continue

            time_until = start_time - now
            minutes_until = time_until.total_seconds() / 60

            signals.append(self.create_signal(
                title=f"Upcoming: {event.get('title', 'Event')}",
                content=self._format_event_content(event, minutes_until),
                signal_type=SignalType.CALENDAR_EVENT,
                raw_data={
                    "event": event,
                    "minutes_until": minutes_until,
                }
            ))

        return signals

    def get_free_slots(
        self,
        date: datetime = None,
        min_duration_minutes: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Find free time slots in the calendar.

        Args:
            date: Date to check (defaults to today)
            min_duration_minutes: Minimum slot duration

        Returns:
            List of free time slot dictionaries
        """
        # This would require more sophisticated calendar analysis
        # Placeholder for future implementation
        return []

    def _validate_config(self) -> List[str]:
        """Validate calendar watcher configuration."""
        errors = []

        lookahead = self.get_setting("lookahead_days", 7)
        if not isinstance(lookahead, int) or lookahead < 1:
            errors.append("lookahead_days must be a positive integer")

        reminder_minutes = self.get_setting("reminder_minutes", [])
        if not isinstance(reminder_minutes, list):
            errors.append("reminder_minutes must be a list")
        elif not all(isinstance(m, (int, float)) and m > 0 for m in reminder_minutes):
            errors.append("reminder_minutes must contain positive numbers")

        return errors
