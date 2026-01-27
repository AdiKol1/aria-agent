# Aria v4.0 - Always-On Reach Implementation Plan

> **Objective**: Transform Aria from a Claude Code-bound assistant to an always-on, proactively reachable AI companion.
>
> **Features**:
> 1. Daemon Mode (Background Service)
> 2. Scheduled Tasks (Cron for Briefings)
> 3. Push Notifications (ntfy.sh)
> 4. Telegram Bridge (Messaging Access)
>
> **Estimated Scope**: ~1500 lines of new code across 8 new files + modifications to 6 existing files

---

## Multi-Agent Execution Protocol

This plan is designed for parallel execution by multiple Claude Code agents. Each feature is **independent** and can be developed simultaneously.

### Agent Assignment

| Agent | Feature | Primary Files | Dependencies |
|-------|---------|---------------|--------------|
| **Agent A** | Daemon Mode | `aria/daemon.py`, `aria/daemon_api.py` | None |
| **Agent B** | Scheduled Tasks | `aria/scheduler.py` | Daemon (can mock) |
| **Agent C** | Push Notifications | `aria/notifications.py` | None |
| **Agent D** | Telegram Bridge | `aria/bridges/telegram.py` | Notifications |

### Communication Protocol

Agents should update this file with status:
```markdown
## Status Tracking
- [ ] Agent A: Daemon Mode - NOT STARTED
- [ ] Agent B: Scheduled Tasks - NOT STARTED
- [ ] Agent C: Push Notifications - NOT STARTED
- [ ] Agent D: Telegram Bridge - NOT STARTED
```

---

# Feature 1: Daemon Mode (Agent A)

## Goal
Aria runs as a background macOS service (launchd) with a REST API for control, surviving terminal/Claude Code closure.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ARIA DAEMON                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌─────────────────┐    ┌────────────┐ │
│  │   REST API   │    │  Ambient Loop   │    │  Scheduler │ │
│  │   (uvicorn)  │◄──►│  (existing)     │◄──►│  (new)     │ │
│  │   port 8420  │    │                 │    │            │ │
│  └──────────────┘    └─────────────────┘    └────────────┘ │
│         │                    │                    │         │
│         ▼                    ▼                    ▼         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Shared Components                        │  │
│  │  MCP Server │ Memory │ Control │ Vision │ Voice      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
         │
         │ HTTP/WebSocket
         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Claude Code    │    │  Telegram Bot   │    │  Mobile App     │
│  (MCP Client)   │    │  (Bridge)       │    │  (Future)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Files to Create

### 1.1 `aria/daemon.py` (~250 lines)

```python
"""
Aria Daemon - Background Service

Runs Aria as a persistent background service with:
- REST API for external control
- WebSocket for real-time updates
- Ambient intelligence loop
- Scheduled task execution
"""

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import ARIA_HOME, DATA_PATH
from .ambient import get_ambient_system
from .mcp_server import AriaMCPServer

logger = logging.getLogger(__name__)

# Daemon state
PID_FILE = ARIA_HOME / "daemon.pid"
LOG_FILE = ARIA_HOME / "daemon.log"
SOCKET_FILE = ARIA_HOME / "daemon.sock"

# Global instances
_daemon: Optional["AriaDaemon"] = None


class AriaDaemon:
    """
    Main daemon controller.

    Manages:
    - FastAPI server for REST/WebSocket API
    - Ambient intelligence loop
    - Scheduler for cron jobs
    - Notification delivery
    """

    def __init__(self, port: int = 8420):
        self.port = port
        self.app = self._create_app()
        self.mcp_server = AriaMCPServer()
        self.ambient = None
        self.scheduler = None
        self.running = False
        self._websocket_clients: list[WebSocket] = []

    def _create_app(self) -> FastAPI:
        """Create FastAPI application."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info("Aria daemon starting...")
            await self._startup()
            yield
            # Shutdown
            logger.info("Aria daemon stopping...")
            await self._shutdown()

        app = FastAPI(
            title="Aria Daemon API",
            version="4.0.0",
            lifespan=lifespan
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Import and include routers
        from .daemon_api import router
        app.include_router(router)

        # WebSocket endpoint
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._websocket_clients.append(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    # Handle incoming WebSocket messages
                    response = await self._handle_ws_message(data)
                    await websocket.send_json(response)
            except WebSocketDisconnect:
                self._websocket_clients.remove(websocket)

        return app

    async def _startup(self):
        """Initialize daemon components."""
        # Write PID file
        PID_FILE.write_text(str(os.getpid()))

        # Start ambient system
        self.ambient = get_ambient_system()
        await self.ambient.start()

        # Start scheduler
        from .scheduler import get_scheduler
        self.scheduler = get_scheduler()
        await self.scheduler.start()

        self.running = True
        logger.info(f"Aria daemon running on port {self.port}")

    async def _shutdown(self):
        """Cleanup on shutdown."""
        self.running = False

        # Stop scheduler
        if self.scheduler:
            await self.scheduler.stop()

        # Stop ambient system
        if self.ambient:
            await self.ambient.stop()

        # Remove PID file
        if PID_FILE.exists():
            PID_FILE.unlink()

    async def _handle_ws_message(self, data: str) -> dict:
        """Handle WebSocket message."""
        import json
        try:
            msg = json.loads(data)
            action = msg.get("action", "")

            if action == "ping":
                return {"status": "pong"}
            elif action == "status":
                return self.get_status()
            elif action == "briefing":
                return {"briefing": self.ambient.get_briefing()}
            else:
                return {"error": f"Unknown action: {action}"}
        except Exception as e:
            return {"error": str(e)}

    async def broadcast(self, message: dict):
        """Broadcast message to all WebSocket clients."""
        import json
        for client in self._websocket_clients:
            try:
                await client.send_json(message)
            except:
                pass

    def get_status(self) -> dict:
        """Get daemon status."""
        return {
            "running": self.running,
            "pid": os.getpid(),
            "port": self.port,
            "ambient_running": self.ambient.is_running if self.ambient else False,
            "scheduler_running": self.scheduler.is_running if self.scheduler else False,
            "websocket_clients": len(self._websocket_clients),
        }

    def run(self):
        """Run the daemon."""
        uvicorn.run(
            self.app,
            host="127.0.0.1",
            port=self.port,
            log_level="info",
        )


def get_daemon() -> AriaDaemon:
    """Get or create daemon instance."""
    global _daemon
    if _daemon is None:
        _daemon = AriaDaemon()
    return _daemon


def is_daemon_running() -> bool:
    """Check if daemon is already running."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)  # Check if process exists
        return True
    except (ProcessLookupError, ValueError):
        PID_FILE.unlink()
        return False


def start_daemon(foreground: bool = False):
    """Start the daemon."""
    if is_daemon_running():
        print("Aria daemon is already running")
        return

    if foreground:
        # Run in foreground (for debugging)
        daemon = get_daemon()
        daemon.run()
    else:
        # Daemonize
        import subprocess
        subprocess.Popen(
            [sys.executable, "-m", "aria.daemon", "--foreground"],
            stdout=open(LOG_FILE, "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"Aria daemon started. Log: {LOG_FILE}")


def stop_daemon():
    """Stop the daemon."""
    if not PID_FILE.exists():
        print("Aria daemon is not running")
        return

    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        print("Aria daemon stopped")
    except ProcessLookupError:
        print("Daemon process not found")
        PID_FILE.unlink()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()

    if args.stop:
        stop_daemon()
    else:
        start_daemon(foreground=args.foreground)
```

### 1.2 `aria/daemon_api.py` (~200 lines)

```python
"""
Aria Daemon REST API

Endpoints for controlling Aria externally.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

router = APIRouter(prefix="/api/v1", tags=["aria"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ProcessRequest(BaseModel):
    """Request to process natural language input."""
    text: str
    include_screen: bool = False
    context: Optional[Dict[str, Any]] = None


class MemoryRequest(BaseModel):
    """Request to remember or recall."""
    fact: Optional[str] = None
    query: Optional[str] = None
    category: str = "other"


class ScheduleRequest(BaseModel):
    """Request to schedule a task."""
    task_type: str  # "briefing", "research", "reminder"
    schedule: str   # cron expression or time like "08:00"
    payload: Dict[str, Any] = {}


class NotificationRequest(BaseModel):
    """Request to send a notification."""
    title: str
    body: str
    priority: str = "default"  # "min", "low", "default", "high", "urgent"
    tags: List[str] = []


# ============================================================================
# STATUS ENDPOINTS
# ============================================================================

@router.get("/status")
async def get_status():
    """Get daemon status."""
    from .daemon import get_daemon
    return get_daemon().get_status()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# ============================================================================
# PROCESSING ENDPOINTS
# ============================================================================

@router.post("/process")
async def process_request(request: ProcessRequest):
    """Process a natural language request."""
    from .daemon import get_daemon
    from .agent import get_agent

    agent = get_agent()
    response = agent.process_request(
        request.text,
        include_screen=request.include_screen
    )
    return {"response": response}


@router.post("/command")
async def execute_command(request: ProcessRequest):
    """Execute an ambient command."""
    from .daemon import get_daemon
    daemon = get_daemon()

    # Process through ambient system
    result = daemon.mcp_server._process_ambient_command(request.text)
    return {"result": result}


# ============================================================================
# MEMORY ENDPOINTS
# ============================================================================

@router.post("/memory/remember")
async def remember(request: MemoryRequest):
    """Store a fact in memory."""
    from .memory import get_memory

    if not request.fact:
        raise HTTPException(400, "fact is required")

    memory = get_memory()
    success = memory.remember_fact(request.fact, request.category)
    return {"success": success, "fact": request.fact}


@router.post("/memory/recall")
async def recall(request: MemoryRequest):
    """Search memory."""
    from .memory import get_memory

    if not request.query:
        raise HTTPException(400, "query is required")

    memory = get_memory()
    results = memory.recall_facts(request.query, n_results=10)
    return {"results": results}


@router.get("/memory/all")
async def list_memories():
    """List all memories."""
    from .memory import get_memory
    memory = get_memory()
    return {"memories": memory.get_all_facts()}


# ============================================================================
# AMBIENT ENDPOINTS
# ============================================================================

@router.get("/ambient/briefing")
async def get_briefing(format: str = "text"):
    """Get current briefing."""
    from .daemon import get_daemon
    daemon = get_daemon()

    if daemon.ambient:
        return {"briefing": daemon.ambient.get_briefing(format=format)}
    return {"error": "Ambient system not running"}


@router.get("/ambient/worlds")
async def list_worlds():
    """List all worlds."""
    from .daemon import get_daemon
    daemon = get_daemon()

    if daemon.ambient:
        worlds = daemon.ambient.list_worlds()
        return {"worlds": [
            {"id": w.id, "name": w.name, "description": w.description}
            for w in worlds
        ]}
    return {"error": "Ambient system not running"}


@router.get("/ambient/insights")
async def get_insights(limit: int = 10, priority: Optional[str] = None):
    """Get pending insights."""
    from .daemon import get_daemon
    daemon = get_daemon()

    if daemon.ambient:
        insights = daemon.ambient.get_pending_insights(limit=limit, priority=priority)
        return {"insights": [
            {"id": i.id, "title": i.title, "priority": i.priority.value}
            for i in insights
        ]}
    return {"error": "Ambient system not running"}


# ============================================================================
# SCHEDULER ENDPOINTS
# ============================================================================

@router.get("/schedule")
async def list_scheduled_tasks():
    """List all scheduled tasks."""
    from .scheduler import get_scheduler
    scheduler = get_scheduler()
    return {"tasks": scheduler.list_tasks()}


@router.post("/schedule")
async def create_scheduled_task(request: ScheduleRequest):
    """Create a scheduled task."""
    from .scheduler import get_scheduler
    scheduler = get_scheduler()

    task_id = scheduler.add_task(
        task_type=request.task_type,
        schedule=request.schedule,
        payload=request.payload
    )
    return {"task_id": task_id, "success": True}


@router.delete("/schedule/{task_id}")
async def delete_scheduled_task(task_id: str):
    """Delete a scheduled task."""
    from .scheduler import get_scheduler
    scheduler = get_scheduler()

    success = scheduler.remove_task(task_id)
    if not success:
        raise HTTPException(404, "Task not found")
    return {"success": True}


# ============================================================================
# NOTIFICATION ENDPOINTS
# ============================================================================

@router.post("/notify")
async def send_notification(request: NotificationRequest, background_tasks: BackgroundTasks):
    """Send a push notification."""
    from .notifications import get_notifier
    notifier = get_notifier()

    background_tasks.add_task(
        notifier.send,
        title=request.title,
        body=request.body,
        priority=request.priority,
        tags=request.tags
    )
    return {"queued": True}


@router.get("/notify/test")
async def test_notification():
    """Send a test notification."""
    from .notifications import get_notifier
    notifier = get_notifier()

    success = await notifier.send(
        title="Aria Test",
        body="Notifications are working!",
        priority="default"
    )
    return {"success": success}
```

### 1.3 `com.aria.daemon.plist` (Update existing)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.aria.daemon</string>

    <key>ProgramArguments</key>
    <array>
        <string>/Users/adikol/aria-agent/venv/bin/python</string>
        <string>-m</string>
        <string>aria.daemon</string>
        <string>--foreground</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/adikol/aria-agent</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/Users/adikol/aria-agent/venv/bin:/usr/local/bin:/usr/bin:/bin</string>
        <key>PYTHONPATH</key>
        <string>/Users/adikol/aria-agent</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>StandardOutPath</key>
    <string>/Users/adikol/.aria/daemon.log</string>

    <key>StandardErrorPath</key>
    <string>/Users/adikol/.aria/daemon.error.log</string>

    <key>ThrottleInterval</key>
    <integer>10</integer>
</dict>
</plist>
```

## Files to Modify

### 1.4 `aria/config.py` - Add daemon configuration

```python
# Add after line 27 (DATA_PATH)

# Daemon Configuration
DAEMON_PORT = int(os.getenv("ARIA_DAEMON_PORT", "8420"))
DAEMON_HOST = os.getenv("ARIA_DAEMON_HOST", "127.0.0.1")
DAEMON_AUTO_START = os.getenv("ARIA_DAEMON_AUTO_START", "false").lower() == "true"
```

### 1.5 Update `requirements.txt`

```
# Add these dependencies
fastapi>=0.109.0
uvicorn>=0.27.0
python-multipart>=0.0.6
websockets>=12.0
httpx>=0.26.0
```

## Installation Script

### 1.6 `install_daemon.sh`

```bash
#!/bin/bash
# Install Aria Daemon as a macOS service

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST_SRC="$SCRIPT_DIR/com.aria.daemon.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.aria.daemon.plist"

echo "Installing Aria Daemon..."

# Copy plist
cp "$PLIST_SRC" "$PLIST_DST"

# Load the service
launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load "$PLIST_DST"

echo "Aria Daemon installed and started."
echo "Check status: launchctl list | grep aria"
echo "View logs: tail -f ~/.aria/daemon.log"
```

## Success Criteria

- [ ] `python -m aria.daemon` starts daemon on port 8420
- [ ] `curl http://localhost:8420/api/v1/status` returns status
- [ ] Daemon survives terminal closure
- [ ] launchd restarts daemon on crash
- [ ] WebSocket connections work for real-time updates

---

# Feature 2: Scheduled Tasks (Agent B)

## Goal
Cron-like scheduler for automated briefings, research, and reminders integrated with the ambient system.

## Files to Create

### 2.1 `aria/scheduler.py` (~350 lines)

```python
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
from datetime import datetime, time as dt_time, timedelta
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
    days_of_week: List[int] = field(default_factory=lambda: [0,1,2,3,4])  # Mon-Fri

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
            days_of_week=data.get("days_of_week", [0,1,2,3,4]),
        )

    def calculate_next_run(self) -> datetime:
        """Calculate next run time."""
        now = datetime.now()

        if self.frequency == TaskFrequency.ONCE:
            # Parse as datetime
            return datetime.fromisoformat(self.schedule)

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
            for _ in range(7):
                if next_time > now and next_time.weekday() in self.days_of_week:
                    return next_time
                next_time += timedelta(days=1)
            return next_time

        elif self.frequency == TaskFrequency.HOURLY:
            # Run at :MM every hour
            minute = int(self.schedule)
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
            days_of_week=days_of_week or [0,1,2,3,4],
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
        asyncio.create_task(self._run_loop())

    async def stop(self):
        """Stop the scheduler."""
        self._running = False
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
            body=briefing[:500],  # Truncate for notification
            priority=task.payload.get("priority", "default"),
            tags=["briefing"]
        )

        logger.info("Briefing delivered")

    async def _handle_research(self, task: ScheduledTask):
        """Handle research task."""
        from .learning_engine import get_learning_engine
        from .memory import get_memory

        learning = get_learning_engine()
        memory = get_memory()

        # Run research
        topic = task.payload.get("topic", "general")
        query = task.payload.get("query", topic)

        # Use web search if available, otherwise skip
        try:
            from .tools import web_search
            result = web_search(query)
            if result:
                memory.remember_fact(
                    f"Research on {topic} ({datetime.now().strftime('%Y-%m-%d')}): {result[:500]}",
                    category="research"
                )
                logger.info(f"Research completed: {topic}")
        except Exception as e:
            logger.warning(f"Research failed: {e}")

    async def _handle_reminder(self, task: ScheduledTask):
        """Handle reminder task."""
        from .notifications import get_notifier

        notifier = get_notifier()

        await notifier.send(
            title=task.payload.get("title", "Reminder"),
            body=task.payload.get("body", "You have a reminder"),
            priority=task.payload.get("priority", "high"),
            tags=["reminder"]
        )

    async def _handle_world_update(self, task: ScheduledTask):
        """Handle world update task."""
        from .ambient import get_ambient_system

        ambient = get_ambient_system()

        # Force a cycle to update worlds
        if ambient.is_running:
            await ambient.run_cycle()
            logger.info("World update completed")

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

def schedule_morning_briefing(time: str = "08:00"):
    """Schedule a daily morning briefing."""
    scheduler = get_scheduler()
    return scheduler.add_task(
        task_type="briefing",
        schedule=time,
        payload={"title": "Morning Briefing", "format": "text"},
        frequency="daily",
    )


def schedule_evening_briefing(time: str = "18:00"):
    """Schedule a daily evening briefing."""
    scheduler = get_scheduler()
    return scheduler.add_task(
        task_type="briefing",
        schedule=time,
        payload={"title": "Evening Summary", "format": "text"},
        frequency="daily",
    )


def schedule_research(topic: str, query: str, frequency: str = "weekly"):
    """Schedule periodic research on a topic."""
    scheduler = get_scheduler()
    return scheduler.add_task(
        task_type="research",
        schedule="09:00",
        payload={"topic": topic, "query": query},
        frequency=frequency,
    )


def schedule_reminder(title: str, body: str, at: str, repeat: str = "once"):
    """Schedule a reminder."""
    scheduler = get_scheduler()
    return scheduler.add_task(
        task_type="reminder",
        schedule=at,
        payload={"title": title, "body": body},
        frequency=repeat,
    )
```

## MCP Server Integration

### 2.2 Add to `aria/mcp_server.py` (Add these tools)

```python
# Add to get_tools() list:

# Scheduling
{
    "name": "schedule_briefing",
    "description": "Schedule automatic briefings. Examples: 'brief me every morning at 8am', 'evening summary at 6pm'",
    "inputSchema": {
        "type": "object",
        "properties": {
            "time": {"type": "string", "description": "Time in HH:MM format (e.g., '08:00')"},
            "type": {"type": "string", "enum": ["morning", "evening"], "default": "morning"},
            "days": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Days of week (0=Mon, 6=Sun). Default: weekdays"
            }
        },
        "required": ["time"]
    }
},
{
    "name": "schedule_reminder",
    "description": "Schedule a reminder notification.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Reminder title"},
            "body": {"type": "string", "description": "Reminder message"},
            "at": {"type": "string", "description": "Time or datetime (e.g., '14:00' or '2026-01-26T14:00')"},
            "repeat": {"type": "string", "enum": ["once", "daily", "weekly"], "default": "once"}
        },
        "required": ["title", "body", "at"]
    }
},
{
    "name": "list_scheduled_tasks",
    "description": "List all scheduled tasks (briefings, reminders, research).",
    "inputSchema": {
        "type": "object",
        "properties": {},
        "required": []
    }
},
{
    "name": "cancel_scheduled_task",
    "description": "Cancel a scheduled task by ID.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "ID of the task to cancel"}
        },
        "required": ["task_id"]
    }
},

# Add to call_tool() handlers:

elif name == "schedule_briefing":
    from .scheduler import get_scheduler
    scheduler = get_scheduler()

    time = arguments["time"]
    briefing_type = arguments.get("type", "morning")
    days = arguments.get("days", [0,1,2,3,4])  # Weekdays

    task_id = scheduler.add_task(
        task_type="briefing",
        schedule=time,
        payload={
            "title": f"{briefing_type.title()} Briefing",
            "format": "text"
        },
        frequency="daily",
        days_of_week=days
    )
    return {"content": [{"type": "text", "text": f"Scheduled {briefing_type} briefing at {time}. Task ID: {task_id}"}]}

elif name == "schedule_reminder":
    from .scheduler import schedule_reminder

    task_id = schedule_reminder(
        title=arguments["title"],
        body=arguments["body"],
        at=arguments["at"],
        repeat=arguments.get("repeat", "once")
    )
    return {"content": [{"type": "text", "text": f"Reminder scheduled. Task ID: {task_id}"}]}

elif name == "list_scheduled_tasks":
    from .scheduler import get_scheduler
    scheduler = get_scheduler()
    tasks = scheduler.list_tasks()

    if tasks:
        lines = [f"Scheduled tasks ({len(tasks)}):"]
        for t in tasks:
            status = "enabled" if t["enabled"] else "disabled"
            lines.append(f"- [{t['id']}] {t['task_type']} at {t['schedule']} ({status})")
            if t.get("next_run"):
                lines.append(f"  Next: {t['next_run']}")
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}
    return {"content": [{"type": "text", "text": "No scheduled tasks."}]}

elif name == "cancel_scheduled_task":
    from .scheduler import get_scheduler
    scheduler = get_scheduler()

    success = scheduler.remove_task(arguments["task_id"])
    if success:
        return {"content": [{"type": "text", "text": f"Cancelled task {arguments['task_id']}"}]}
    return {"content": [{"type": "text", "text": f"Task not found: {arguments['task_id']}"}]}
```

## Success Criteria

- [ ] `schedule_morning_briefing("08:00")` creates task
- [ ] Tasks persist across daemon restarts
- [ ] Briefings delivered at scheduled times
- [ ] Tasks can be listed, enabled, disabled, deleted

---

# Feature 3: Push Notifications (Agent C)

## Goal
Send notifications to user's devices via ntfy.sh (self-hostable, free, mobile apps available).

## Files to Create

### 3.1 `aria/notifications.py` (~200 lines)

```python
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

from .config import DATA_PATH

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
            server_url=os.getenv("NTFY_SERVER", "https://ntfy.sh"),
            topic=os.getenv("NTFY_TOPIC", "aria-notifications"),
            auth_token=os.getenv("NTFY_TOKEN"),
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

    async def send_briefing(self, briefing: str, priority: str = "default"):
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

    async def send_insight(self, title: str, body: str, priority: str = "high"):
        """Send an insight notification."""
        return await self.send(
            title=f"Aria Insight: {title}",
            body=body,
            priority=priority,
            tags=["bulb", "bell"],
        )

    async def send_reminder(self, title: str, body: str):
        """Send a reminder notification."""
        return await self.send(
            title=title,
            body=body,
            priority="high",
            tags=["alarm_clock"],
        )

    async def send_alert(self, title: str, body: str):
        """Send an urgent alert."""
        return await self.send(
            title=f"⚠️ {title}",
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
    priority = priority_map.get(str(action.priority), "default")

    return await notifier.send(title=title, body=body, priority=priority)
```

### 3.2 Add to `aria/config.py`

```python
# Add after DAEMON config

# Notification Configuration
NTFY_SERVER = os.getenv("NTFY_SERVER", "https://ntfy.sh")
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "")  # User must configure
NTFY_TOKEN = os.getenv("NTFY_TOKEN", "")
```

### 3.3 Add to `.env.example`

```bash
# Push Notifications (ntfy.sh)
# Get your own topic at https://ntfy.sh or self-host
NTFY_SERVER=https://ntfy.sh
NTFY_TOPIC=your-unique-topic-here
NTFY_TOKEN=  # Optional: for authentication
```

### 3.4 MCP Server Integration

Add to `aria/mcp_server.py`:

```python
# Add to get_tools():
{
    "name": "configure_notifications",
    "description": "Configure push notifications via ntfy.sh. First time setup required.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "ntfy.sh topic (unique identifier)"},
            "server_url": {"type": "string", "description": "ntfy server URL (default: https://ntfy.sh)"},
            "test": {"type": "boolean", "default": True, "description": "Send test notification"}
        },
        "required": ["topic"]
    }
},
{
    "name": "send_notification",
    "description": "Send a push notification to user's devices.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Notification title"},
            "body": {"type": "string", "description": "Notification body"},
            "priority": {"type": "string", "enum": ["min", "low", "default", "high", "urgent"], "default": "default"}
        },
        "required": ["title", "body"]
    }
},

# Add to call_tool():
elif name == "configure_notifications":
    from .notifications import get_notifier
    import asyncio

    notifier = get_notifier()
    notifier.configure(
        topic=arguments["topic"],
        server_url=arguments.get("server_url", "https://ntfy.sh")
    )

    if arguments.get("test", True):
        success = asyncio.run(notifier.send(
            title="Aria Notifications Configured",
            body="You'll receive notifications here from now on!",
            priority="default"
        ))
        return {"content": [{"type": "text", "text": f"Notifications configured. Test sent: {'success' if success else 'failed'}"}]}
    return {"content": [{"type": "text", "text": "Notifications configured."}]}

elif name == "send_notification":
    from .notifications import get_notifier
    import asyncio

    notifier = get_notifier()
    success = asyncio.run(notifier.send(
        title=arguments["title"],
        body=arguments["body"],
        priority=arguments.get("priority", "default")
    ))
    return {"content": [{"type": "text", "text": f"Notification {'sent' if success else 'failed'}"}]}
```

## Success Criteria

- [ ] `notifier.send("Test", "Body")` sends notification
- [ ] Notifications appear on iOS/Android ntfy app
- [ ] Configuration persists
- [ ] Integration with DeliveryEngine works

---

# Feature 4: Telegram Bridge (Agent D)

## Goal
Access Aria from Telegram with full functionality - process requests, get briefings, receive insights.

## Files to Create

### 4.1 `aria/bridges/__init__.py`

```python
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
```

### 4.2 `aria/bridges/telegram.py` (~400 lines)

```python
"""
Aria Telegram Bridge

Access Aria from Telegram with full functionality.

Features:
- Natural language processing
- Ambient commands (worlds, entities, goals)
- Briefings on demand
- Receive insights as they happen
- Voice message support (future)
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from telegram import Update, Bot
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from ..config import DATA_PATH

logger = logging.getLogger(__name__)

# Configuration
TELEGRAM_CONFIG_FILE = DATA_PATH / "telegram_config.json"


class TelegramBridge:
    """
    Telegram bot that bridges to Aria.

    Usage:
        bridge = TelegramBridge()
        await bridge.start()
    """

    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.allowed_user_ids: List[int] = []  # Security: only allowed users
        self.app: Optional[Application] = None
        self._running = False

        self._load_config()

    def _load_config(self):
        """Load config from file."""
        if TELEGRAM_CONFIG_FILE.exists():
            try:
                with open(TELEGRAM_CONFIG_FILE) as f:
                    data = json.load(f)
                    self.bot_token = data.get("bot_token", self.bot_token)
                    self.allowed_user_ids = data.get("allowed_user_ids", [])
            except Exception as e:
                logger.warning(f"Failed to load Telegram config: {e}")

    def _save_config(self):
        """Save config to file."""
        try:
            with open(TELEGRAM_CONFIG_FILE, "w") as f:
                json.dump({
                    "bot_token": self.bot_token,
                    "allowed_user_ids": self.allowed_user_ids,
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save Telegram config: {e}")

    def configure(self, bot_token: str = None, user_id: int = None):
        """Configure the bridge."""
        if bot_token:
            self.bot_token = bot_token
        if user_id and user_id not in self.allowed_user_ids:
            self.allowed_user_ids.append(user_id)
        self._save_config()

    def _is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized."""
        # If no users configured, allow anyone (first-time setup)
        if not self.allowed_user_ids:
            return True
        return user_id in self.allowed_user_ids

    async def start(self):
        """Start the Telegram bot."""
        if not self.bot_token:
            logger.error("Telegram bot token not configured")
            return

        self.app = Application.builder().token(self.bot_token).build()

        # Register handlers
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("help", self._cmd_help))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("briefing", self._cmd_briefing))
        self.app.add_handler(CommandHandler("worlds", self._cmd_worlds))
        self.app.add_handler(CommandHandler("insights", self._cmd_insights))
        self.app.add_handler(CommandHandler("remember", self._cmd_remember))
        self.app.add_handler(CommandHandler("recall", self._cmd_recall))
        self.app.add_handler(CommandHandler("notify", self._cmd_notify))

        # Handle all other messages
        self.app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._handle_message
        ))

        self._running = True
        logger.info("Telegram bridge starting...")

        # Start polling
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)

        logger.info("Telegram bridge running")

    async def stop(self):
        """Stop the Telegram bot."""
        if self.app:
            self._running = False
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            logger.info("Telegram bridge stopped")

    # =========================================================================
    # COMMAND HANDLERS
    # =========================================================================

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user = update.effective_user

        # First-time setup: register user
        if not self.allowed_user_ids:
            self.allowed_user_ids.append(user.id)
            self._save_config()
            await update.message.reply_text(
                f"Hi {user.first_name}! I'm Aria, your AI assistant.\n\n"
                f"You're now registered as an authorized user.\n\n"
                f"Try these commands:\n"
                f"/briefing - Get your daily briefing\n"
                f"/worlds - List your monitored domains\n"
                f"/insights - Get pending insights\n"
                f"/help - See all commands\n\n"
                f"Or just send me a message!"
            )
        elif self._is_authorized(user.id):
            await update.message.reply_text(
                f"Welcome back, {user.first_name}!\n"
                f"Send /help to see available commands."
            )
        else:
            await update.message.reply_text(
                "Sorry, you're not authorized to use this bot."
            )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        if not self._is_authorized(update.effective_user.id):
            return

        help_text = """
*Aria Commands*

*Information*
/status - Daemon and system status
/briefing - Get your current briefing
/worlds - List monitored worlds
/insights - Get pending insights

*Memory*
/remember <fact> - Store a fact
/recall <query> - Search memory

*Notifications*
/notify <message> - Send yourself a notification

*Natural Language*
Just send any message to interact naturally!
Examples:
- "What's going on?"
- "Track Compass as a competitor"
- "My goal is to close 3 deals"
- "Schedule morning briefing at 8am"
        """
        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        if not self._is_authorized(update.effective_user.id):
            return

        from ..daemon import is_daemon_running, get_daemon

        if is_daemon_running():
            daemon = get_daemon()
            status = daemon.get_status()

            text = (
                f"*Aria Status*\n\n"
                f"Daemon: Running ✅\n"
                f"Port: {status['port']}\n"
                f"Ambient: {'Running' if status['ambient_running'] else 'Stopped'}\n"
                f"Scheduler: {'Running' if status['scheduler_running'] else 'Stopped'}\n"
                f"WebSocket clients: {status['websocket_clients']}"
            )
        else:
            text = "*Aria Status*\n\nDaemon: Not running ❌"

        await update.message.reply_text(text, parse_mode="Markdown")

    async def _cmd_briefing(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /briefing command."""
        if not self._is_authorized(update.effective_user.id):
            return

        from ..ambient import get_ambient_system

        ambient = get_ambient_system()
        briefing = ambient.get_briefing(format="text")

        # Split if too long for Telegram (4096 char limit)
        if len(briefing) > 4000:
            parts = [briefing[i:i+4000] for i in range(0, len(briefing), 4000)]
            for i, part in enumerate(parts):
                await update.message.reply_text(
                    f"*Briefing ({i+1}/{len(parts)})*\n\n{part}",
                    parse_mode="Markdown"
                )
        else:
            await update.message.reply_text(
                f"*Briefing*\n\n{briefing}",
                parse_mode="Markdown"
            )

    async def _cmd_worlds(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /worlds command."""
        if not self._is_authorized(update.effective_user.id):
            return

        from ..ambient import get_ambient_system

        ambient = get_ambient_system()
        worlds = ambient.list_worlds()

        if worlds:
            lines = ["*Your Worlds*\n"]
            for world in worlds:
                status = "🟢" if world.is_active_now() else "⚪"
                lines.append(f"{status} *{world.name}*")
                lines.append(f"   {world.description}")
                lines.append(f"   Goals: {len(world.goals)}, Entities: {len(world.entities)}")
            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        else:
            await update.message.reply_text(
                "No worlds configured yet.\n\n"
                "Create one by saying something like:\n"
                "\"I work in real estate\""
            )

    async def _cmd_insights(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /insights command."""
        if not self._is_authorized(update.effective_user.id):
            return

        from ..ambient import get_ambient_system

        ambient = get_ambient_system()
        insights = ambient.get_pending_insights(limit=5)

        if insights:
            lines = ["*Pending Insights*\n"]
            for insight in insights:
                priority_emoji = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🟢",
                }.get(insight.priority.value, "⚪")
                lines.append(f"{priority_emoji} *{insight.title}*")
                if insight.suggested_action:
                    lines.append(f"   → {insight.suggested_action}")
            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        else:
            await update.message.reply_text("No pending insights. 🎉")

    async def _cmd_remember(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /remember command."""
        if not self._is_authorized(update.effective_user.id):
            return

        fact = " ".join(context.args) if context.args else ""

        if not fact:
            await update.message.reply_text("Usage: /remember <fact to remember>")
            return

        from ..memory import get_memory
        memory = get_memory()
        success = memory.remember_fact(fact, "telegram")

        if success:
            await update.message.reply_text(f"✅ Remembered: {fact}")
        else:
            await update.message.reply_text("❌ Failed to remember")

    async def _cmd_recall(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /recall command."""
        if not self._is_authorized(update.effective_user.id):
            return

        query = " ".join(context.args) if context.args else ""

        if not query:
            await update.message.reply_text("Usage: /recall <search query>")
            return

        from ..memory import get_memory
        memory = get_memory()
        results = memory.recall_facts(query, n_results=5)

        if results:
            lines = ["*Memories found:*\n"]
            for r in results:
                lines.append(f"• {r['fact']}")
            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        else:
            await update.message.reply_text("No memories found for that query.")

    async def _cmd_notify(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /notify command (test notifications)."""
        if not self._is_authorized(update.effective_user.id):
            return

        message = " ".join(context.args) if context.args else "Test notification from Aria"

        from ..notifications import get_notifier
        notifier = get_notifier()
        success = await notifier.send("Aria", message)

        if success:
            await update.message.reply_text("✅ Notification sent")
        else:
            await update.message.reply_text("❌ Notification failed (check ntfy config)")

    # =========================================================================
    # MESSAGE HANDLER
    # =========================================================================

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle natural language messages."""
        if not self._is_authorized(update.effective_user.id):
            return

        text = update.message.text

        # Send typing indicator
        await update.message.chat.send_action("typing")

        try:
            # Process through MCP server (same as Claude Code)
            from ..mcp_server import AriaMCPServer
            server = AriaMCPServer()

            # Try ambient command first
            result = server._process_ambient_command(text)

            # If not an ambient command, try general processing
            if "didn't understand" in result.lower():
                from ..agent import get_agent
                agent = get_agent()
                result = agent.process_request(text, include_screen=False)

            # Send response
            await update.message.reply_text(result)

        except Exception as e:
            logger.error(f"Message handling error: {e}")
            await update.message.reply_text(
                "Sorry, I encountered an error processing that request."
            )

    # =========================================================================
    # PROACTIVE MESSAGING
    # =========================================================================

    async def send_insight(self, insight, user_id: int = None):
        """Send an insight to user(s)."""
        if not self.app or not self._running:
            return

        # Send to specified user or all allowed users
        recipients = [user_id] if user_id else self.allowed_user_ids

        for uid in recipients:
            try:
                priority_emoji = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🟢",
                }.get(str(insight.priority), "⚪")

                text = (
                    f"{priority_emoji} *New Insight*\n\n"
                    f"*{insight.title}*\n\n"
                    f"{insight.body}\n\n"
                )
                if insight.suggested_action:
                    text += f"_Suggested: {insight.suggested_action}_"

                await self.app.bot.send_message(uid, text, parse_mode="Markdown")
            except Exception as e:
                logger.error(f"Failed to send insight to {uid}: {e}")

    async def send_briefing(self, briefing: str, user_id: int = None):
        """Send a briefing to user(s)."""
        if not self.app or not self._running:
            return

        recipients = [user_id] if user_id else self.allowed_user_ids

        for uid in recipients:
            try:
                await self.app.bot.send_message(
                    uid,
                    f"*Daily Briefing*\n\n{briefing}",
                    parse_mode="Markdown"
                )
            except Exception as e:
                logger.error(f"Failed to send briefing to {uid}: {e}")

    # =========================================================================
    # STATUS
    # =========================================================================

    @property
    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "configured": bool(self.bot_token),
            "authorized_users": len(self.allowed_user_ids),
        }
```

### 4.3 Add to `requirements.txt`

```
python-telegram-bot>=21.0
```

### 4.4 Add to `.env.example`

```bash
# Telegram Bridge
# Create a bot via @BotFather on Telegram
TELEGRAM_BOT_TOKEN=your-bot-token-here
```

### 4.5 Add to `aria/config.py`

```python
# Telegram Bridge
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN)
```

### 4.6 MCP Server Integration

Add to `aria/mcp_server.py`:

```python
# Add to get_tools():
{
    "name": "configure_telegram",
    "description": "Configure Telegram bridge for messaging access to Aria.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "bot_token": {"type": "string", "description": "Telegram bot token from @BotFather"}
        },
        "required": ["bot_token"]
    }
},
{
    "name": "telegram_status",
    "description": "Get Telegram bridge status.",
    "inputSchema": {
        "type": "object",
        "properties": {},
        "required": []
    }
},

# Add to call_tool():
elif name == "configure_telegram":
    from .bridges import get_telegram_bridge
    bridge = get_telegram_bridge()
    bridge.configure(bot_token=arguments["bot_token"])
    return {"content": [{"type": "text", "text": "Telegram bridge configured. Start the daemon to activate."}]}

elif name == "telegram_status":
    from .bridges import get_telegram_bridge
    bridge = get_telegram_bridge()
    status = bridge.get_status()
    lines = [
        "Telegram Bridge Status:",
        f"  Running: {status['running']}",
        f"  Configured: {status['configured']}",
        f"  Authorized users: {status['authorized_users']}",
    ]
    return {"content": [{"type": "text", "text": "\n".join(lines)}]}
```

### 4.7 Daemon Integration

Add to `aria/daemon.py` in `_startup()`:

```python
# Start Telegram bridge if configured
from .config import TELEGRAM_ENABLED
if TELEGRAM_ENABLED:
    from .bridges import get_telegram_bridge
    self.telegram = get_telegram_bridge()
    await self.telegram.start()
    logger.info("Telegram bridge started")
```

## Success Criteria

- [ ] Bot responds to /start command
- [ ] /briefing returns current briefing
- [ ] Natural language messages are processed
- [ ] Insights can be pushed to Telegram
- [ ] Only authorized users can access

---

# Integration Points

## Update `aria/ambient/delivery/engine.py`

Register notification handler:

```python
# In DeliveryEngine.__init__():
from ..notifications import notification_delivery_handler
self.register_handler(DeliveryChannel.PUSH_NOTIFICATION, notification_delivery_handler)
```

## Update `CLAUDE.md` (Add to MCP Tools section)

```markdown
### Daemon Control
- `daemon_status` - Check if daemon is running
- `start_daemon` - Start Aria daemon (if not running)

### Scheduling
- `schedule_briefing(time, type)` - Schedule automatic briefings
- `schedule_reminder(title, body, at)` - Schedule reminders
- `list_scheduled_tasks()` - List all scheduled tasks
- `cancel_scheduled_task(task_id)` - Cancel a task

### Notifications
- `configure_notifications(topic)` - Set up ntfy.sh notifications
- `send_notification(title, body)` - Send a push notification

### Telegram
- `configure_telegram(bot_token)` - Set up Telegram bridge
- `telegram_status` - Check Telegram bridge status
```

---

# Testing Checklist

## Feature 1: Daemon Mode
- [ ] `python -m aria.daemon` starts successfully
- [ ] REST API accessible at localhost:8420
- [ ] WebSocket connections work
- [ ] Daemon survives terminal closure
- [ ] launchd auto-starts on login
- [ ] launchd restarts on crash

## Feature 2: Scheduled Tasks
- [ ] Morning briefing scheduled for 08:00
- [ ] Tasks persist across daemon restart
- [ ] Briefing delivered at scheduled time
- [ ] Research tasks execute and store to memory
- [ ] Reminders trigger notifications

## Feature 3: Push Notifications
- [ ] ntfy.sh configuration stored
- [ ] Test notification received on phone
- [ ] Briefings sent via notification
- [ ] Insights trigger notifications
- [ ] Priority levels work correctly

## Feature 4: Telegram Bridge
- [ ] Bot created and token configured
- [ ] /start registers user
- [ ] /briefing returns briefing
- [ ] Natural language processed
- [ ] Insights pushed to Telegram

---

# Rollout Plan

## Phase 1: Foundation (Agent A)
1. Implement daemon.py and daemon_api.py
2. Test REST API
3. Create launchd plist
4. Test auto-start and crash recovery

## Phase 2: Notifications (Agent C)
1. Implement notifications.py
2. Test ntfy.sh integration
3. Integrate with DeliveryEngine
4. Add MCP tools

## Phase 3: Scheduling (Agent B)
1. Implement scheduler.py
2. Test task persistence
3. Integrate with daemon
4. Add MCP tools

## Phase 4: Telegram (Agent D)
1. Implement telegram.py
2. Test bot commands
3. Integrate with daemon
4. Test proactive messaging

## Phase 5: Integration Testing
1. End-to-end: Schedule briefing → Notification → Telegram
2. Stress test: Multiple scheduled tasks
3. Recovery test: Kill daemon, verify restart
4. Security test: Unauthorized Telegram access

---

# Status Tracking

Update this section as work progresses:

## Agent Status
- [x] **Agent A (Daemon)**: COMPLETED - aria/daemon.py, aria/daemon_api.py
- [x] **Agent B (Scheduler)**: COMPLETED - aria/scheduler.py
- [x] **Agent C (Notifications)**: COMPLETED - aria/notifications.py
- [x] **Agent D (Telegram)**: COMPLETED - aria/bridges/telegram.py

## Blockers
- None

## Completed
- [x] Plan created
- [x] aria/daemon.py - Main daemon controller with FastAPI/uvicorn
- [x] aria/daemon_api.py - REST API endpoints
- [x] aria/scheduler.py - Cron-like task scheduling
- [x] aria/notifications.py - Push notifications via ntfy.sh
- [x] aria/bridges/__init__.py - Bridges module init
- [x] aria/bridges/telegram.py - Telegram bot bridge
- [x] aria/config.py - Updated with daemon/notification/telegram config
- [x] requirements.txt - Added fastapi, uvicorn, httpx, python-telegram-bot
- [x] .env.example - Added new environment variables
- [x] com.aria.daemon.plist - launchd service definition
- [x] install_daemon.sh - Installation script

## Pending
- [x] MCP server integration for new tools (schedule_briefing, configure_notifications, etc.)
- [ ] Integration testing
- [ ] Update CLAUDE.md with new tools documentation

---

*Plan created: 2026-01-25*
*Last updated: 2026-01-25*
*Implementation completed: 2026-01-25*
