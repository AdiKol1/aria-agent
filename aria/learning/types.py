"""
Learning System Types

Data structures for skill learning, pattern recognition, and memory management.
Ported from aria-memory TypeScript implementation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# SKILL LEARNING TYPES
# =============================================================================

class ActionType(Enum):
    """Types of actions that can be recorded and replayed."""
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    TYPE = "type"
    SCROLL = "scroll"
    HOTKEY = "hotkey"
    KEY_PRESS = "key_press"
    WAIT = "wait"
    OPEN_APP = "open_app"
    OPEN_URL = "open_url"
    CUSTOM = "custom"


@dataclass
class VisualTarget:
    """
    Visual context for an action - what the user clicked on.

    This enables adaptive replay: if the UI shifts, we can find
    the element by its visual appearance rather than just coordinates.
    """
    description: str  # "blue submit button", "search icon"
    screenshot_region: Optional[str] = None  # Base64 of cropped area around target
    element_text: Optional[str] = None  # Text content if any
    element_type: Optional[str] = None  # "button", "input", "link", etc.
    relative_position: Optional[str] = None  # "top-right", "center", etc.


@dataclass
class RecordedAction:
    """
    A single recorded action within a skill.

    Each action captures:
    - What was done (type, coordinates, text, etc.)
    - Visual context (what it looked like when clicked)
    - Timing (how long the user waited)
    """
    action_type: ActionType
    timestamp: datetime = field(default_factory=datetime.now)

    # Position-based actions
    x: Optional[int] = None
    y: Optional[int] = None

    # Text-based actions
    text: Optional[str] = None

    # Hotkey/key press
    keys: Optional[List[str]] = None

    # Scroll
    scroll_amount: Optional[int] = None

    # Visual context for adaptive replay
    visual_target: Optional[VisualTarget] = None

    # App context
    app_name: Optional[str] = None
    window_title: Optional[str] = None

    # Timing
    delay_before_ms: int = 0  # How long user waited before this action

    # Metadata
    notes: Optional[str] = None  # User annotation
    is_decision_point: bool = False  # User made a choice here

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "action_type": self.action_type.value,
            "timestamp": self.timestamp.isoformat(),
            "x": self.x,
            "y": self.y,
            "text": self.text,
            "keys": self.keys,
            "scroll_amount": self.scroll_amount,
            "visual_target": {
                "description": self.visual_target.description,
                "screenshot_region": self.visual_target.screenshot_region,
                "element_text": self.visual_target.element_text,
                "element_type": self.visual_target.element_type,
                "relative_position": self.visual_target.relative_position,
            } if self.visual_target else None,
            "app_name": self.app_name,
            "window_title": self.window_title,
            "delay_before_ms": self.delay_before_ms,
            "notes": self.notes,
            "is_decision_point": self.is_decision_point,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecordedAction":
        """Create from dictionary."""
        visual_target = None
        if data.get("visual_target"):
            vt = data["visual_target"]
            visual_target = VisualTarget(
                description=vt.get("description", ""),
                screenshot_region=vt.get("screenshot_region"),
                element_text=vt.get("element_text"),
                element_type=vt.get("element_type"),
                relative_position=vt.get("relative_position"),
            )

        return cls(
            action_type=ActionType(data["action_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            x=data.get("x"),
            y=data.get("y"),
            text=data.get("text"),
            keys=data.get("keys"),
            scroll_amount=data.get("scroll_amount"),
            visual_target=visual_target,
            app_name=data.get("app_name"),
            window_title=data.get("window_title"),
            delay_before_ms=data.get("delay_before_ms", 0),
            notes=data.get("notes"),
            is_decision_point=data.get("is_decision_point", False),
        )


@dataclass
class LearnedSkill:
    """
    A complete learned skill that can be replayed.

    Skills are sequences of actions with:
    - Trigger phrases that invoke them
    - Success criteria to verify completion
    - Adaptations for different contexts
    """
    id: str
    name: str
    description: str

    # When to trigger this skill
    trigger_phrases: List[str] = field(default_factory=list)

    # The actions to perform
    actions: List[RecordedAction] = field(default_factory=list)

    # Decision points - where user input might be needed
    decision_points: List[int] = field(default_factory=list)  # Action indices

    # Success criteria
    success_criteria: Optional[str] = None  # Description of what success looks like
    success_screenshot_region: Optional[str] = None  # What screen should look like

    # Context requirements
    required_app: Optional[str] = None
    required_url_pattern: Optional[str] = None

    # Adaptations learned from multiple executions
    adaptations: Dict[str, Any] = field(default_factory=dict)

    # Statistics
    times_executed: int = 0
    times_succeeded: int = 0
    last_executed: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Learning metadata
    learned_from_user: bool = True  # vs pre-programmed
    confidence: float = 0.5

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.times_executed == 0:
            return 0.0
        return self.times_succeeded / self.times_executed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "trigger_phrases": self.trigger_phrases,
            "actions": [a.to_dict() for a in self.actions],
            "decision_points": self.decision_points,
            "success_criteria": self.success_criteria,
            "success_screenshot_region": self.success_screenshot_region,
            "required_app": self.required_app,
            "required_url_pattern": self.required_url_pattern,
            "adaptations": self.adaptations,
            "times_executed": self.times_executed,
            "times_succeeded": self.times_succeeded,
            "last_executed": self.last_executed.isoformat() if self.last_executed else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "learned_from_user": self.learned_from_user,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearnedSkill":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            trigger_phrases=data.get("trigger_phrases", []),
            actions=[RecordedAction.from_dict(a) for a in data.get("actions", [])],
            decision_points=data.get("decision_points", []),
            success_criteria=data.get("success_criteria"),
            success_screenshot_region=data.get("success_screenshot_region"),
            required_app=data.get("required_app"),
            required_url_pattern=data.get("required_url_pattern"),
            adaptations=data.get("adaptations", {}),
            times_executed=data.get("times_executed", 0),
            times_succeeded=data.get("times_succeeded", 0),
            last_executed=datetime.fromisoformat(data["last_executed"]) if data.get("last_executed") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
            learned_from_user=data.get("learned_from_user", True),
            confidence=data.get("confidence", 0.5),
        )


@dataclass
class RecordingSession:
    """An active skill recording session."""
    id: str
    name: str
    started_at: datetime = field(default_factory=datetime.now)
    actions: List[RecordedAction] = field(default_factory=list)

    # Context at start
    starting_app: Optional[str] = None
    starting_url: Optional[str] = None

    # User annotations
    notes: List[str] = field(default_factory=list)

    # Recording state
    is_paused: bool = False
    last_action_time: Optional[datetime] = None


@dataclass
class ExecutionContext:
    """Context for executing a learned skill."""
    skill: LearnedSkill
    variables: Dict[str, Any] = field(default_factory=dict)  # User-provided values

    # Current state
    current_action_index: int = 0
    started_at: Optional[datetime] = None

    # Adaptive execution
    coordinate_adjustments: Dict[int, tuple] = field(default_factory=dict)  # action_idx -> (dx, dy)

    # Error handling
    retry_count: int = 0
    max_retries: int = 3
    last_error: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of executing a skill."""
    success: bool
    skill_id: str

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # Execution details
    actions_completed: int = 0
    total_actions: int = 0

    # Errors
    error: Optional[str] = None
    failed_at_action: Optional[int] = None

    # Adaptations made during execution
    adaptations_used: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> Optional[int]:
        """Calculate execution duration in milliseconds."""
        if not self.completed_at:
            return None
        delta = self.completed_at - self.started_at
        return int(delta.total_seconds() * 1000)


# =============================================================================
# PATTERN LEARNING TYPES
# =============================================================================

class ObservationType(Enum):
    """Types of observations that can become patterns."""
    CORRECTION = "correction"  # User corrected Aria's action
    REPEATED_ACTION = "repeated_action"  # User does this repeatedly
    CONSISTENT_CHOICE = "consistent_choice"  # User always picks this option
    FAILURE_RECOVERY = "failure_recovery"  # User fixed something Aria broke


@dataclass
class Observation:
    """
    A single observation that might become a pattern.

    Observations are collected over time. When enough similar
    observations accumulate, they're promoted to a pattern.
    """
    id: str
    observation_type: ObservationType
    timestamp: datetime = field(default_factory=datetime.now)

    # What happened
    description: str = ""

    # For corrections: what was wrong and what was right
    original_action: Optional[str] = None
    corrected_action: Optional[str] = None

    # Context
    context: Dict[str, Any] = field(default_factory=dict)  # app, url, task, etc.

    # Related observations
    similar_to: List[str] = field(default_factory=list)  # Other observation IDs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "observation_type": self.observation_type.value,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "original_action": self.original_action,
            "corrected_action": self.corrected_action,
            "context": self.context,
            "similar_to": self.similar_to,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            observation_type=ObservationType(data["observation_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            description=data.get("description", ""),
            original_action=data.get("original_action"),
            corrected_action=data.get("corrected_action"),
            context=data.get("context", {}),
            similar_to=data.get("similar_to", []),
        )


@dataclass
class LearnedPattern:
    """
    A pattern learned from multiple observations.

    Patterns are rules that Aria follows based on observed user behavior.
    Example: "When copying code, always use code block formatting"
    """
    id: str

    # The pattern itself
    trigger: str  # When this pattern applies
    action: str  # What to do

    # Context constraints
    context: Dict[str, Any] = field(default_factory=dict)

    # Evidence that led to this pattern
    evidence: List[str] = field(default_factory=list)  # Observation IDs
    observation_count: int = 0

    # Confidence and performance
    confidence: float = 0.5
    times_applied: int = 0
    times_successful: int = 0

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)
    last_applied: Optional[datetime] = None
    is_archived: bool = False

    # Learning metadata
    auto_generated: bool = True  # vs explicitly taught
    failure_modes: List[str] = field(default_factory=list)  # Known failure cases

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.times_applied == 0:
            return 0.0
        return self.times_successful / self.times_applied

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "trigger": self.trigger,
            "action": self.action,
            "context": self.context,
            "evidence": self.evidence,
            "observation_count": self.observation_count,
            "confidence": self.confidence,
            "times_applied": self.times_applied,
            "times_successful": self.times_successful,
            "created_at": self.created_at.isoformat(),
            "last_applied": self.last_applied.isoformat() if self.last_applied else None,
            "is_archived": self.is_archived,
            "auto_generated": self.auto_generated,
            "failure_modes": self.failure_modes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearnedPattern":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            trigger=data["trigger"],
            action=data["action"],
            context=data.get("context", {}),
            evidence=data.get("evidence", []),
            observation_count=data.get("observation_count", 0),
            confidence=data.get("confidence", 0.5),
            times_applied=data.get("times_applied", 0),
            times_successful=data.get("times_successful", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            last_applied=datetime.fromisoformat(data["last_applied"]) if data.get("last_applied") else None,
            is_archived=data.get("is_archived", False),
            auto_generated=data.get("auto_generated", True),
            failure_modes=data.get("failure_modes", []),
        )


# =============================================================================
# MEMORY PRUNING TYPES
# =============================================================================

class MemoryType(Enum):
    """Types of memories with different pruning policies."""
    PREFERENCE = "preference"  # User preferences - never auto-decay
    FACT = "fact"  # Facts about user - flag on contradiction
    PATTERN = "pattern"  # Learned patterns - archive if unused/poor performance
    INSIGHT = "insight"  # Time-sensitive insights - decay after 7 days
    SKILL = "skill"  # Learned skills - archive if consistently failing
    INTERACTION = "interaction"  # Past conversations - oldest first


@dataclass
class PruningPolicy:
    """Policy for how to prune a memory type."""
    memory_type: MemoryType

    # Decay settings
    auto_decay: bool = False
    decay_days: int = 30  # Days of inactivity before considering decay

    # Performance thresholds
    min_success_rate: float = 0.0  # Below this, consider archiving
    min_usage_count: int = 0  # Must be used at least this many times

    # Actions
    archive_on_decay: bool = True  # Archive vs hard delete
    flag_on_contradiction: bool = False  # Flag conflicting info instead of overwriting

    # Special rules
    never_auto_delete: bool = False  # Requires explicit user action

    @classmethod
    def for_type(cls, memory_type: MemoryType) -> "PruningPolicy":
        """Get the default policy for a memory type."""
        policies = {
            MemoryType.PREFERENCE: cls(
                memory_type=memory_type,
                auto_decay=False,
                never_auto_delete=True,
            ),
            MemoryType.FACT: cls(
                memory_type=memory_type,
                auto_decay=False,
                flag_on_contradiction=True,
                never_auto_delete=True,
            ),
            MemoryType.PATTERN: cls(
                memory_type=memory_type,
                auto_decay=True,
                decay_days=30,
                min_success_rate=0.3,
                archive_on_decay=True,
            ),
            MemoryType.INSIGHT: cls(
                memory_type=memory_type,
                auto_decay=True,
                decay_days=7,
                archive_on_decay=True,
            ),
            MemoryType.SKILL: cls(
                memory_type=memory_type,
                auto_decay=True,
                decay_days=60,
                min_success_rate=0.2,
                min_usage_count=3,
                archive_on_decay=True,
            ),
            MemoryType.INTERACTION: cls(
                memory_type=memory_type,
                auto_decay=True,
                decay_days=90,
                archive_on_decay=False,  # Can delete old interactions
            ),
        }
        return policies.get(memory_type, cls(memory_type=memory_type))
