"""
Aria Scheduler - Cron-like Task Scheduling

Executes scheduled tasks:
- Morning/evening briefings
- Proactive research
- Custom reminders
- Periodic world updates
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
import hashlib

from .config import DATA_PATH

logger = logging.getLogger(__name__)

# Storage
SCHEDULE_FILE = DATA_PATH / "scheduled_tasks.json"


class TaskType(Enum):
    BRIEFING = "briefing"
    RESEARCH = "research"
    REMINDER = "reminder"
    WORLD_UPDATE = "world_update"
    CUSTOM = "custom"


class TaskFrequency(Enum):
    ONCE = "once"
    DAILY = "daily"
    WEEKLY = "weekly"
    HOURLY = "hourly"
    CRON = "cron"


@dataclass
class ScheduledTask:
    """A scheduled task definition."""
    id: str
    task_type: TaskType
    frequency: TaskFrequency
    schedule: str  # Time like "08:00" or cron expression
    payload: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # For weekly tasks
    days_of_week: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_type": self.task_type.value,
            "frequency": self.frequency.value,
            "schedule": self.schedule,
            "payload": self.payload,
            "enabled": self.enabled,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "created_at": self.created_at,
            "days_of_week": self.days_of_week,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScheduledTask":
        return cls(
            id=data["id"],
            task_type=TaskType(data["task_type"]),
            frequency=TaskFrequency(data["frequency"]),
            schedule=data["schedule"],
            payload=data.get("payload", {}),
            enabled=data.get("enabled", True),
            last_run=data.get("last_run"),
            next_run=data.get("next_run"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            days_of_week=data.get("days_of_week", [0, 1, 2, 3, 4]),
        )

    def calculate_next_run(self) -> datetime:
        """Calculate next run time."""
        now = datetime.now()

        if self.frequency == TaskFrequency.ONCE:
            # Parse as datetime
            try:
                return datetime.fromisoformat(self.schedule)
            except ValueError:
                # Maybe it's just a time, use today
                hour, minute = map(int, self.schedule.split(":"))
                return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        elif self.frequency == TaskFrequency.DAILY:
            # Parse as time HH:MM
            hour, minute = map(int, self.schedule.split(":"))
            next_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_time <= now:
                next_time += timedelta(days=1)
            return next_time

        elif self.frequency == TaskFrequency.WEEKLY:
            hour, minute = map(int, self.schedule.split(":"))
            next_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

            # Find next valid day
            for _ in range(8):
                if next_time > now and next_time.weekday() in self.days_of_week:
                    return next_time
                next_time += timedelta(days=1)
            return next_time

        elif self.frequency == TaskFrequency.HOURLY:
            # Run at :MM every hour
            minute = int(self.schedule.split(":")[-1]) if ":" in self.schedule else int(self.schedule)
            next_time = now.replace(minute=minute, second=0, microsecond=0)
            if next_time <= now:
                next_time += timedelta(hours=1)
            return next_time

        # Default: 1 hour from now
        return now + timedelta(hours=1)


class Scheduler:
    """
    Task scheduler with persistence.

    Runs scheduled tasks at their specified times.
    """

    def __init__(self):
        self._tasks: Dict[str, ScheduledTask] = {}
        self._handlers: Dict[TaskType, Callable] = {}
        self._running = False
        self._check_interval = 30  # Check every 30 seconds
        self._loop_task: Optional[asyncio.Task] = None

        self._load_tasks()
        self._register_default_handlers()

    def _load_tasks(self):
        """Load tasks from disk."""
        if SCHEDULE_FILE.exists():
            try:
                with open(SCHEDULE_FILE) as f:
                    data = json.load(f)
                    for task_data in data.get("tasks", []):
                        task = ScheduledTask.from_dict(task_data)
                        self._tasks[task.id] = task
                logger.info(f"Loaded {len(self._tasks)} scheduled tasks")
            except Exception as e:
                logger.error(f"Failed to load tasks: {e}")

    def _save_tasks(self):
        """Save tasks to disk."""
        try:
            data = {
                "tasks": [t.to_dict() for t in self._tasks.values()],
                "updated_at": datetime.now().isoformat()
            }
            with open(SCHEDULE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tasks: {e}")

    def _register_default_handlers(self):
        """Register built-in task handlers."""
        self._handlers[TaskType.BRIEFING] = self._handle_briefing
        self._handlers[TaskType.RESEARCH] = self._handle_research
        self._handlers[TaskType.REMINDER] = self._handle_reminder
        self._handlers[TaskType.WORLD_UPDATE] = self._handle_world_update

    # =========================================================================
    # TASK MANAGEMENT
    # =========================================================================

    def add_task(
        self,
        task_type: str,
        schedule: str,
        payload: Dict[str, Any] = None,
        frequency: str = "daily",
        days_of_week: List[int] = None,
    ) -> str:
        """Add a new scheduled task."""
        task_id = hashlib.md5(
            f"{task_type}{schedule}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        task = ScheduledTask(
            id=task_id,
            task_type=TaskType(task_type),
            frequency=TaskFrequency(frequency),
            schedule=schedule,
            payload=payload or {},
            days_of_week=days_of_week or [0, 1, 2, 3, 4],
        )

        task.next_run = task.calculate_next_run().isoformat()

        self._tasks[task_id] = task
        self._save_tasks()

        logger.info(f"Added task {task_id}: {task_type} at {schedule}")
        return task_id

    def remove_task(self, task_id: str) -> bool:
        """Remove a scheduled task."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            self._save_tasks()
            return True
        return False

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[dict]:
        """List all tasks."""
        return [t.to_dict() for t in self._tasks.values()]

    def enable_task(self, task_id: str, enabled: bool = True) -> bool:
        """Enable or disable a task."""
        if task_id in self._tasks:
            self._tasks[task_id].enabled = enabled
            self._save_tasks()
            return True
        return False

    # =========================================================================
    # EXECUTION LOOP
    # =========================================================================

    async def start(self):
        """Start the scheduler."""
        if self._running:
            return

        self._running = True
        logger.info("Scheduler started")

        # Start the check loop
        self._loop_task = asyncio.create_task(self._run_loop())

    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")

    async def _run_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                await self._check_and_run_tasks()
            except Exception as e:
                logger.error(f"Scheduler error: {e}")

            await asyncio.sleep(self._check_interval)

    async def _check_and_run_tasks(self):
        """Check for due tasks and execute them."""
        now = datetime.now()

        for task in list(self._tasks.values()):
            if not task.enabled:
                continue

            if not task.next_run:
                task.next_run = task.calculate_next_run().isoformat()
                continue

            next_run = datetime.fromisoformat(task.next_run)

            if now >= next_run:
                logger.info(f"Running task: {task.id} ({task.task_type.value})")

                try:
                    await self._execute_task(task)
                    task.last_run = now.isoformat()

                    # Calculate next run (unless it's a one-time task)
                    if task.frequency == TaskFrequency.ONCE:
                        task.enabled = False
                    else:
                        task.next_run = task.calculate_next_run().isoformat()

                    self._save_tasks()

                except Exception as e:
                    logger.error(f"Task {task.id} failed: {e}")

    async def _execute_task(self, task: ScheduledTask):
        """Execute a single task."""
        handler = self._handlers.get(task.task_type)

        if not handler:
            logger.warning(f"No handler for task type: {task.task_type}")
            return

        if asyncio.iscoroutinefunction(handler):
            await handler(task)
        else:
            handler(task)

    # =========================================================================
    # TASK HANDLERS
    # =========================================================================

    async def _handle_briefing(self, task: ScheduledTask):
        """Handle briefing task."""
        try:
            from .ambient import get_ambient_system
            from .notifications import get_notifier

            ambient = get_ambient_system()
            notifier = get_notifier()

            # Generate briefing
            format_type = task.payload.get("format", "text")
            briefing = ambient.get_briefing(format=format_type)

            # Send notification
            await notifier.send(
                title=f"Aria {task.payload.get('title', 'Briefing')}",
                body=briefing[:500] if briefing else "No updates available.",  # Truncate for notification
                priority=task.payload.get("priority", "default"),
                tags=["briefing"]
            )

            logger.info("Briefing delivered")
        except Exception as e:
            logger.error(f"Briefing handler error: {e}")

    async def _handle_research(self, task: ScheduledTask):
        """Handle research task."""
        try:
            from .memory import get_memory

            memory = get_memory()

            # Run research
            topic = task.payload.get("topic", "general")
            query = task.payload.get("query", topic)

            # TODO: Implement web search when available
            logger.info(f"Research task for topic: {topic} (query: {query})")

            # Store research note
            memory.remember_fact(
                f"Research scheduled for {topic} ({datetime.now().strftime('%Y-%m-%d')})",
                category="research"
            )
        except Exception as e:
            logger.error(f"Research handler error: {e}")

    async def _handle_reminder(self, task: ScheduledTask):
        """Handle reminder task."""
        try:
            from .notifications import get_notifier

            notifier = get_notifier()

            await notifier.send(
                title=task.payload.get("title", "Reminder"),
                body=task.payload.get("body", "You have a reminder"),
                priority=task.payload.get("priority", "high"),
                tags=["reminder", "alarm_clock"]
            )

            logger.info("Reminder delivered")
        except Exception as e:
            logger.error(f"Reminder handler error: {e}")

    async def _handle_world_update(self, task: ScheduledTask):
        """Handle world update task."""
        try:
            from .ambient import get_ambient_system

            ambient = get_ambient_system()

            # Force a cycle to update worlds
            if ambient.is_running:
                await ambient.run_cycle()
                logger.info("World update completed")
        except Exception as e:
            logger.error(f"World update handler error: {e}")

    # =========================================================================
    # STATUS
    # =========================================================================

    @property
    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> dict:
        """Get scheduler status."""
        return {
            "running": self._running,
            "task_count": len(self._tasks),
            "enabled_count": sum(1 for t in self._tasks.values() if t.enabled),
            "check_interval": self._check_interval,
        }


# Singleton
_scheduler: Optional[Scheduler] = None


def get_scheduler() -> Scheduler:
    """Get singleton scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = Scheduler()
    return _scheduler


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def schedule_morning_briefing(time: str = "08:00") -> str:
    """Schedule a daily morning briefing."""
    scheduler = get_scheduler()
    return scheduler.add_task(
        task_type="briefing",
        schedule=time,
        payload={"title": "Morning Briefing", "format": "text"},
        frequency="daily",
    )


def schedule_evening_briefing(time: str = "18:00") -> str:
    """Schedule a daily evening briefing."""
    scheduler = get_scheduler()
    return scheduler.add_task(
        task_type="briefing",
        schedule=time,
        payload={"title": "Evening Summary", "format": "text"},
        frequency="daily",
    )


def schedule_research(topic: str, query: str, frequency: str = "weekly") -> str:
    """Schedule periodic research on a topic."""
    scheduler = get_scheduler()
    return scheduler.add_task(
        task_type="research",
        schedule="09:00",
        payload={"topic": topic, "query": query},
        frequency=frequency,
    )


def schedule_reminder(title: str, body: str, at: str, repeat: str = "once") -> str:
    """Schedule a reminder."""
    scheduler = get_scheduler()
    return scheduler.add_task(
        task_type="reminder",
        schedule=at,
        payload={"title": title, "body": body},
        frequency=repeat,
    )
