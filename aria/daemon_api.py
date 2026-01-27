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
    frequency: str = "daily"  # once, daily, weekly, hourly
    payload: Dict[str, Any] = {}
    days_of_week: Optional[List[int]] = None  # 0=Mon, 6=Sun


class NotificationRequest(BaseModel):
    """Request to send a notification."""
    title: str
    body: str
    priority: str = "default"  # "min", "low", "default", "high", "urgent"
    tags: List[str] = []


class ReminderRequest(BaseModel):
    """Request to create a reminder."""
    title: str
    body: str
    at: str  # Time like "14:00" or datetime "2026-01-26T14:00"
    repeat: str = "once"  # once, daily, weekly


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
    try:
        from .agent import get_agent

        agent = get_agent()
        response = agent.process_request(
            request.text,
            include_screen=request.include_screen
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/command")
async def execute_command(request: ProcessRequest):
    """Execute an ambient command."""
    from .daemon import get_daemon
    daemon = get_daemon()

    if daemon.mcp_server:
        result = daemon.mcp_server._process_ambient_command(request.text)
        return {"result": result}
    raise HTTPException(status_code=503, detail="MCP server not available")


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
        briefing = daemon.ambient.get_briefing(format=format)
        return {"briefing": briefing}
    raise HTTPException(status_code=503, detail="Ambient system not running")


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
    raise HTTPException(status_code=503, detail="Ambient system not running")


@router.get("/ambient/insights")
async def get_insights(limit: int = 10, priority: Optional[str] = None):
    """Get pending insights."""
    from .daemon import get_daemon
    daemon = get_daemon()

    if daemon.ambient:
        insights = daemon.ambient.get_pending_insights(limit=limit, priority=priority)
        return {"insights": [
            {
                "id": i.id,
                "title": i.title,
                "body": i.body,
                "priority": i.priority.value if hasattr(i.priority, 'value') else str(i.priority)
            }
            for i in insights
        ]}
    raise HTTPException(status_code=503, detail="Ambient system not running")


# ============================================================================
# SCHEDULER ENDPOINTS
# ============================================================================

@router.get("/schedule")
async def list_scheduled_tasks():
    """List all scheduled tasks."""
    try:
        from .scheduler import get_scheduler
        scheduler = get_scheduler()
        return {"tasks": scheduler.list_tasks()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/schedule")
async def create_scheduled_task(request: ScheduleRequest):
    """Create a scheduled task."""
    try:
        from .scheduler import get_scheduler
        scheduler = get_scheduler()

        task_id = scheduler.add_task(
            task_type=request.task_type,
            schedule=request.schedule,
            payload=request.payload,
            frequency=request.frequency,
            days_of_week=request.days_of_week,
        )
        return {"task_id": task_id, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/schedule/{task_id}")
async def delete_scheduled_task(task_id: str):
    """Delete a scheduled task."""
    from .scheduler import get_scheduler
    scheduler = get_scheduler()

    success = scheduler.remove_task(task_id)
    if not success:
        raise HTTPException(404, "Task not found")
    return {"success": True}


@router.post("/schedule/briefing")
async def schedule_briefing(time: str, briefing_type: str = "morning", days: Optional[List[int]] = None):
    """Schedule a daily briefing."""
    try:
        from .scheduler import get_scheduler
        scheduler = get_scheduler()

        task_id = scheduler.add_task(
            task_type="briefing",
            schedule=time,
            payload={
                "title": f"{briefing_type.title()} Briefing",
                "format": "text"
            },
            frequency="daily",
            days_of_week=days or [0, 1, 2, 3, 4],  # Weekdays
        )
        return {"task_id": task_id, "message": f"Scheduled {briefing_type} briefing at {time}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schedule/reminder")
async def schedule_reminder(request: ReminderRequest):
    """Schedule a reminder."""
    try:
        from .scheduler import schedule_reminder

        task_id = schedule_reminder(
            title=request.title,
            body=request.body,
            at=request.at,
            repeat=request.repeat
        )
        return {"task_id": task_id, "message": "Reminder scheduled"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# NOTIFICATION ENDPOINTS
# ============================================================================

@router.post("/notify")
async def send_notification(request: NotificationRequest, background_tasks: BackgroundTasks):
    """Send a push notification."""
    try:
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
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/notify/test")
async def test_notification():
    """Send a test notification."""
    try:
        from .notifications import get_notifier
        notifier = get_notifier()

        success = await notifier.send(
            title="Aria Test",
            body="Notifications are working!",
            priority="default"
        )
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/notify/status")
async def notification_status():
    """Get notification configuration status."""
    try:
        from .notifications import get_notifier
        notifier = get_notifier()
        return notifier.get_status()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# ============================================================================
# TELEGRAM ENDPOINTS
# ============================================================================

@router.get("/telegram/status")
async def telegram_status():
    """Get Telegram bridge status."""
    from .daemon import get_daemon
    daemon = get_daemon()

    if daemon.telegram:
        return daemon.telegram.get_status()
    return {"running": False, "configured": False, "authorized_users": 0}


# ============================================================================
# VOICE ENDPOINTS
# ============================================================================

@router.post("/speak")
async def speak(text: str):
    """Speak text aloud."""
    try:
        from .voice import speak as tts_speak
        tts_speak(text)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# LEARNING ENDPOINTS
# ============================================================================

@router.get("/learning/status")
async def learning_status():
    """Get learning system status."""
    try:
        from .learning_engine import get_learning_engine
        engine = get_learning_engine()
        return {
            "success_rate": engine.get_success_rate(),
            "session_count": len(engine.sessions) if hasattr(engine, 'sessions') else 0,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/learning/skills")
async def list_skills():
    """List learned skills."""
    try:
        from .learning_engine import get_learning_engine
        engine = get_learning_engine()
        skills = engine.list_skills()
        return {"skills": skills}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/learning/patterns")
async def list_patterns():
    """List learned patterns."""
    try:
        from .learning_engine import get_learning_engine
        engine = get_learning_engine()
        patterns = engine.list_patterns()
        return {"patterns": patterns}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
