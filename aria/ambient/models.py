"""
Aria Ambient Intelligence - Core Data Models

Defines all data classes for the ambient system including:
- World: A domain/context the user operates in
- Goal: Something the user wants to achieve
- Entity: A person, company, or thing to track
- Signal: Raw observation from a watcher
- Insight: Synthesized, actionable intelligence
- PreparedAction: Ready-to-execute action
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import json
import hashlib
import uuid

from .constants import (
    SignalType,
    InsightPriority,
    InsightStatus,
    ActionType,
    DeliveryChannel,
    EntityType,
    RelationshipType,
    GoalStatus,
    GoalPriority,
    UrgencyLevel,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_id(prefix: str = "") -> str:
    """Generate a unique ID."""
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}_{uid}" if prefix else uid


def now_iso() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


# =============================================================================
# WORLD MODELS
# =============================================================================

@dataclass
class Schedule:
    """When a world is typically active."""
    active_days: List[str] = field(default_factory=lambda: [
        "monday", "tuesday", "wednesday", "thursday", "friday"
    ])
    active_hours_start: str = "09:00"
    active_hours_end: str = "18:00"
    timezone: str = "local"

    def is_active_now(self) -> bool:
        """Check if schedule is currently active."""
        now = datetime.now()
        day_name = now.strftime("%A").lower()

        if day_name not in self.active_days:
            return False

        try:
            start = datetime.strptime(self.active_hours_start, "%H:%M").time()
            end = datetime.strptime(self.active_hours_end, "%H:%M").time()
            return start <= now.time() <= end
        except ValueError:
            return True  # Default to active if parsing fails

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_days": self.active_days,
            "active_hours_start": self.active_hours_start,
            "active_hours_end": self.active_hours_end,
            "timezone": self.timezone,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Schedule":
        return cls(
            active_days=data.get("active_days", []),
            active_hours_start=data.get("active_hours_start", "09:00"),
            active_hours_end=data.get("active_hours_end", "18:00"),
            timezone=data.get("timezone", "local"),
        )


@dataclass
class Goal:
    """Something the user wants to achieve in a world."""
    id: str = field(default_factory=lambda: generate_id("goal"))
    world_id: str = ""
    description: str = ""
    priority: GoalPriority = GoalPriority.MEDIUM

    # Progress tracking
    progress_indicators: List[str] = field(default_factory=list)
    risk_indicators: List[str] = field(default_factory=list)
    deadline: Optional[str] = None  # ISO format

    # Status
    status: GoalStatus = GoalStatus.ACTIVE
    progress: float = 0.0  # 0.0 to 1.0

    # Metadata
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    notes: str = ""

    def is_deadline_approaching(self, days_threshold: int = 7) -> bool:
        """Check if deadline is within threshold."""
        if not self.deadline:
            return False
        try:
            deadline_dt = datetime.fromisoformat(self.deadline)
            return datetime.now() + timedelta(days=days_threshold) >= deadline_dt
        except ValueError:
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "world_id": self.world_id,
            "description": self.description,
            "priority": self.priority.value,
            "progress_indicators": self.progress_indicators,
            "risk_indicators": self.risk_indicators,
            "deadline": self.deadline,
            "status": self.status.value,
            "progress": self.progress,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Goal":
        return cls(
            id=data.get("id", generate_id("goal")),
            world_id=data.get("world_id", ""),
            description=data.get("description", ""),
            priority=GoalPriority(data.get("priority", "medium")),
            progress_indicators=data.get("progress_indicators", []),
            risk_indicators=data.get("risk_indicators", []),
            deadline=data.get("deadline"),
            status=GoalStatus(data.get("status", "active")),
            progress=data.get("progress", 0.0),
            created_at=data.get("created_at", now_iso()),
            updated_at=data.get("updated_at", now_iso()),
            notes=data.get("notes", ""),
        )


@dataclass
class Entity:
    """A person, company, topic, or thing to track."""
    id: str = field(default_factory=lambda: generate_id("entity"))
    world_id: str = ""
    name: str = ""
    type: EntityType = EntityType.CUSTOM
    relationship: RelationshipType = RelationshipType.WATCH

    # Monitoring
    importance: float = 0.5  # 0.0 to 1.0
    watch_for: List[str] = field(default_factory=list)  # Events to watch for

    # Context
    notes: str = ""
    aliases: List[str] = field(default_factory=list)  # Alternative names
    urls: List[str] = field(default_factory=list)     # Related URLs to monitor

    # Metadata
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    last_activity: Optional[str] = None

    def matches_text(self, text: str) -> bool:
        """Check if this entity is mentioned in text."""
        text_lower = text.lower()
        if self.name.lower() in text_lower:
            return True
        for alias in self.aliases:
            if alias.lower() in text_lower:
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "world_id": self.world_id,
            "name": self.name,
            "type": self.type.value,
            "relationship": self.relationship.value,
            "importance": self.importance,
            "watch_for": self.watch_for,
            "notes": self.notes,
            "aliases": self.aliases,
            "urls": self.urls,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_activity": self.last_activity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(
            id=data.get("id", generate_id("entity")),
            world_id=data.get("world_id", ""),
            name=data.get("name", ""),
            type=EntityType(data.get("type", "custom")),
            relationship=RelationshipType(data.get("relationship", "watch")),
            importance=data.get("importance", 0.5),
            watch_for=data.get("watch_for", []),
            notes=data.get("notes", ""),
            aliases=data.get("aliases", []),
            urls=data.get("urls", []),
            created_at=data.get("created_at", now_iso()),
            updated_at=data.get("updated_at", now_iso()),
            last_activity=data.get("last_activity"),
        )


@dataclass
class World:
    """A domain/context the user operates in."""
    id: str = field(default_factory=lambda: generate_id("world"))
    name: str = ""
    description: str = ""

    # What matters
    goals: List[Goal] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)

    # Signal sources
    information_sources: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    # Timing
    schedule: Optional[Schedule] = None

    # Learned patterns
    successful_approaches: List[str] = field(default_factory=list)
    failure_patterns: List[str] = field(default_factory=list)

    # Metadata
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    last_active: Optional[str] = None

    # Learning confidence
    confidence: float = 0.5  # How well does Aria understand this world?

    def is_active_now(self) -> bool:
        """Check if this world should be active based on schedule."""
        if self.schedule is None:
            return True  # Always active if no schedule
        return self.schedule.is_active_now()

    def get_active_goals(self) -> List[Goal]:
        """Get all active goals."""
        return [g for g in self.goals if g.status == GoalStatus.ACTIVE]

    def get_high_importance_entities(self, threshold: float = 0.7) -> List[Entity]:
        """Get entities above importance threshold."""
        return [e for e in self.entities if e.importance >= threshold]

    def matches_keywords(self, text: str) -> List[str]:
        """Find which keywords match in text."""
        text_lower = text.lower()
        return [kw for kw in self.keywords if kw.lower() in text_lower]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "goals": [g.to_dict() for g in self.goals],
            "entities": [e.to_dict() for e in self.entities],
            "information_sources": self.information_sources,
            "keywords": self.keywords,
            "schedule": self.schedule.to_dict() if self.schedule else None,
            "successful_approaches": self.successful_approaches,
            "failure_patterns": self.failure_patterns,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_active": self.last_active,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "World":
        return cls(
            id=data.get("id", generate_id("world")),
            name=data.get("name", ""),
            description=data.get("description", ""),
            goals=[Goal.from_dict(g) for g in data.get("goals", [])],
            entities=[Entity.from_dict(e) for e in data.get("entities", [])],
            information_sources=data.get("information_sources", []),
            keywords=data.get("keywords", []),
            schedule=Schedule.from_dict(data["schedule"]) if data.get("schedule") else None,
            successful_approaches=data.get("successful_approaches", []),
            failure_patterns=data.get("failure_patterns", []),
            created_at=data.get("created_at", now_iso()),
            updated_at=data.get("updated_at", now_iso()),
            last_active=data.get("last_active"),
            confidence=data.get("confidence", 0.5),
        )


# =============================================================================
# SIGNAL MODELS
# =============================================================================

@dataclass
class WorldMatch:
    """Result of matching a signal to a world."""
    world_id: str
    world_name: str
    relevance_score: float
    matched_entities: List[str] = field(default_factory=list)
    matched_keywords: List[str] = field(default_factory=list)
    matched_goals: List[str] = field(default_factory=list)
    match_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "world_id": self.world_id,
            "world_name": self.world_name,
            "relevance_score": self.relevance_score,
            "matched_entities": self.matched_entities,
            "matched_keywords": self.matched_keywords,
            "matched_goals": self.matched_goals,
            "match_reason": self.match_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldMatch":
        return cls(**data)


@dataclass
class Signal:
    """Raw observation from a watcher."""
    id: str = field(default_factory=lambda: generate_id("sig"))
    source: str = ""          # Which watcher
    type: SignalType = SignalType.CUSTOM

    # Content
    title: str = ""
    content: str = ""
    url: Optional[str] = None

    # Relevance (computed after creation)
    relevant_worlds: List[WorldMatch] = field(default_factory=list)

    # Timing
    timestamp: str = field(default_factory=now_iso)
    expires_at: Optional[str] = None

    # Metadata
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if signal has expired."""
        if not self.expires_at:
            return False
        try:
            expires = datetime.fromisoformat(self.expires_at)
            return datetime.now() > expires
        except ValueError:
            return False

    def get_best_world_match(self) -> Optional[WorldMatch]:
        """Get the highest relevance world match."""
        if not self.relevant_worlds:
            return None
        return max(self.relevant_worlds, key=lambda m: m.relevance_score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "type": self.type.value,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "relevant_worlds": [w.to_dict() for w in self.relevant_worlds],
            "timestamp": self.timestamp,
            "expires_at": self.expires_at,
            "raw_data": self.raw_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signal":
        return cls(
            id=data.get("id", generate_id("sig")),
            source=data.get("source", ""),
            type=SignalType(data.get("type", "custom")),
            title=data.get("title", ""),
            content=data.get("content", ""),
            url=data.get("url"),
            relevant_worlds=[WorldMatch.from_dict(w) for w in data.get("relevant_worlds", [])],
            timestamp=data.get("timestamp", now_iso()),
            expires_at=data.get("expires_at"),
            raw_data=data.get("raw_data", {}),
        )


# =============================================================================
# INSIGHT MODELS
# =============================================================================

@dataclass
class Connection:
    """Cross-world connection between signals/insights."""
    id: str = field(default_factory=lambda: generate_id("conn"))
    world_ids: List[str] = field(default_factory=list)
    description: str = ""
    opportunity: str = ""
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "world_ids": self.world_ids,
            "description": self.description,
            "opportunity": self.opportunity,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Connection":
        return cls(**data)


@dataclass
class Insight:
    """Synthesized, actionable intelligence."""
    id: str = field(default_factory=lambda: generate_id("insight"))
    signal_ids: List[str] = field(default_factory=list)

    # Content
    title: str = ""
    summary: str = ""
    world_id: str = ""

    # Priority
    priority: InsightPriority = InsightPriority.MEDIUM
    priority_score: float = 0.5  # Numeric for sorting
    urgency: UrgencyLevel = UrgencyLevel.WHENEVER

    # Suggested action
    suggested_action: str = ""
    action_type: ActionType = ActionType.NONE

    # Connections
    related_goal_ids: List[str] = field(default_factory=list)
    related_entity_ids: List[str] = field(default_factory=list)
    cross_world_connections: List[Connection] = field(default_factory=list)

    # Status
    status: InsightStatus = InsightStatus.NEW
    prepared_action_id: Optional[str] = None

    # Timing
    created_at: str = field(default_factory=now_iso)
    delivered_at: Optional[str] = None
    acted_at: Optional[str] = None
    expires_at: Optional[str] = None

    def is_cross_world(self) -> bool:
        """Check if this insight spans multiple worlds."""
        return len(self.cross_world_connections) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "signal_ids": self.signal_ids,
            "title": self.title,
            "summary": self.summary,
            "world_id": self.world_id,
            "priority": self.priority.value,
            "priority_score": self.priority_score,
            "urgency": self.urgency.value,
            "suggested_action": self.suggested_action,
            "action_type": self.action_type.value,
            "related_goal_ids": self.related_goal_ids,
            "related_entity_ids": self.related_entity_ids,
            "cross_world_connections": [c.to_dict() for c in self.cross_world_connections],
            "status": self.status.value,
            "prepared_action_id": self.prepared_action_id,
            "created_at": self.created_at,
            "delivered_at": self.delivered_at,
            "acted_at": self.acted_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Insight":
        return cls(
            id=data.get("id", generate_id("insight")),
            signal_ids=data.get("signal_ids", []),
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            world_id=data.get("world_id", ""),
            priority=InsightPriority(data.get("priority", "medium")),
            priority_score=data.get("priority_score", 0.5),
            urgency=UrgencyLevel(data.get("urgency", "whenever")),
            suggested_action=data.get("suggested_action", ""),
            action_type=ActionType(data.get("action_type", "none")),
            related_goal_ids=data.get("related_goal_ids", []),
            related_entity_ids=data.get("related_entity_ids", []),
            cross_world_connections=[Connection.from_dict(c) for c in data.get("cross_world_connections", [])],
            status=InsightStatus(data.get("status", "new")),
            prepared_action_id=data.get("prepared_action_id"),
            created_at=data.get("created_at", now_iso()),
            delivered_at=data.get("delivered_at"),
            acted_at=data.get("acted_at"),
            expires_at=data.get("expires_at"),
        )


# =============================================================================
# ACTION MODELS
# =============================================================================

@dataclass
class QuickAction:
    """A one-click action the user can take."""
    id: str = field(default_factory=lambda: generate_id("action"))
    label: str = ""
    action_type: str = ""  # "send", "schedule", "save", "dismiss"
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "action_type": self.action_type,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuickAction":
        return cls(**data)


@dataclass
class PreparedAction:
    """Ready-to-execute action for an insight."""
    id: str = field(default_factory=lambda: generate_id("prep"))
    insight_id: str = ""

    # What's prepared
    type: ActionType = ActionType.ALERT
    content: str = ""
    options: List[str] = field(default_factory=list)  # Alternative versions

    # Quick actions
    quick_actions: List[QuickAction] = field(default_factory=list)

    # Delivery preferences
    preferred_channel: DeliveryChannel = DeliveryChannel.VOICE_BRIEFING
    delivery_window_start: Optional[str] = None
    delivery_window_end: Optional[str] = None

    # Status
    status: str = "prepared"  # "prepared", "queued", "delivered", "executed", "expired"

    # Timing
    created_at: str = field(default_factory=now_iso)
    delivered_at: Optional[str] = None
    executed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "insight_id": self.insight_id,
            "type": self.type.value,
            "content": self.content,
            "options": self.options,
            "quick_actions": [a.to_dict() for a in self.quick_actions],
            "preferred_channel": self.preferred_channel.value,
            "delivery_window_start": self.delivery_window_start,
            "delivery_window_end": self.delivery_window_end,
            "status": self.status,
            "created_at": self.created_at,
            "delivered_at": self.delivered_at,
            "executed_at": self.executed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreparedAction":
        return cls(
            id=data.get("id", generate_id("prep")),
            insight_id=data.get("insight_id", ""),
            type=ActionType(data.get("type", "alert")),
            content=data.get("content", ""),
            options=data.get("options", []),
            quick_actions=[QuickAction.from_dict(a) for a in data.get("quick_actions", [])],
            preferred_channel=DeliveryChannel(data.get("preferred_channel", "voice")),
            delivery_window_start=data.get("delivery_window_start"),
            delivery_window_end=data.get("delivery_window_end"),
            status=data.get("status", "prepared"),
            created_at=data.get("created_at", now_iso()),
            delivered_at=data.get("delivered_at"),
            executed_at=data.get("executed_at"),
        )


# =============================================================================
# CONSCIOUSNESS MODELS
# =============================================================================

@dataclass
class Thought:
    """A single thought in Aria's consciousness stream."""
    id: str = field(default_factory=lambda: generate_id("thought"))
    content: str = ""
    category: str = ""  # "observation", "concern", "opportunity", "preparation"
    world_id: Optional[str] = None
    related_insight_id: Optional[str] = None
    priority: float = 0.5
    timestamp: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "world_id": self.world_id,
            "related_insight_id": self.related_insight_id,
            "priority": self.priority,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Thought":
        return cls(**data)


@dataclass
class ConsciousnessState:
    """Aria's current mental state."""
    current_focus: List[str] = field(default_factory=list)  # World IDs in focus
    active_thoughts: List[Thought] = field(default_factory=list)
    pending_insights: List[str] = field(default_factory=list)  # Insight IDs
    last_briefing: Optional[str] = None
    updated_at: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_focus": self.current_focus,
            "active_thoughts": [t.to_dict() for t in self.active_thoughts],
            "pending_insights": self.pending_insights,
            "last_briefing": self.last_briefing,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsciousnessState":
        return cls(
            current_focus=data.get("current_focus", []),
            active_thoughts=[Thought.from_dict(t) for t in data.get("active_thoughts", [])],
            pending_insights=data.get("pending_insights", []),
            last_briefing=data.get("last_briefing"),
            updated_at=data.get("updated_at", now_iso()),
        )
