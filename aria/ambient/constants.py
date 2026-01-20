"""
Aria Ambient Intelligence - Constants and Enums

Defines all enums, constants, and configuration values for the ambient system.
"""

from enum import Enum, auto
from typing import Dict, Any


# =============================================================================
# ENUMS
# =============================================================================

class SignalType(Enum):
    """Types of signals that watchers can produce."""
    NEWS_ARTICLE = "news_article"
    SOCIAL_MENTION = "social_mention"
    SOCIAL_POST = "social_post"
    PRICE_CHANGE = "price_change"
    CALENDAR_EVENT = "calendar_event"
    CALENDAR_REMINDER = "calendar_reminder"
    COMPETITOR_ACTIVITY = "competitor_activity"
    TREND_EMERGING = "trend_emerging"
    DEADLINE_APPROACHING = "deadline_approaching"
    ENGAGEMENT_ANOMALY = "engagement_anomaly"
    OPPORTUNITY_DETECTED = "opportunity_detected"
    MARKET_MOVEMENT = "market_movement"
    EMAIL_IMPORTANT = "email_important"
    SCREEN_CONTEXT = "screen_context"
    CUSTOM = "custom"


class InsightPriority(Enum):
    """Priority levels for insights."""
    CRITICAL = "critical"  # Immediate attention required
    HIGH = "high"          # Should see today
    MEDIUM = "medium"      # This week
    LOW = "low"            # Whenever convenient

    @property
    def numeric_value(self) -> float:
        """Get numeric value for calculations."""
        values = {
            "critical": 1.0,
            "high": 0.75,
            "medium": 0.5,
            "low": 0.25
        }
        return values.get(self.value, 0.5)


class InsightStatus(Enum):
    """Status of an insight through its lifecycle."""
    NEW = "new"                    # Just generated
    PREPARING = "preparing"        # Action being prepared
    READY = "ready"                # Ready for delivery
    QUEUED = "queued"              # In delivery queue
    DELIVERED = "delivered"        # Delivered to user
    ACTED = "acted"                # User took action
    DISMISSED = "dismissed"        # User dismissed
    EXPIRED = "expired"            # No longer relevant


class ActionType(Enum):
    """Types of prepared actions."""
    DRAFT_CONTENT = "draft_content"      # Social post, article
    DRAFT_RESPONSE = "draft_response"    # Reply to something
    RESEARCH_BRIEF = "research_brief"    # Research summary
    ALERT = "alert"                      # Simple notification
    SCHEDULE_SUGGESTION = "schedule"     # Calendar suggestion
    TASK_SUGGESTION = "task"             # Todo item
    QUESTION = "question"                # Ask user for input
    NONE = "none"                        # FYI only


class DeliveryChannel(Enum):
    """Channels for delivering insights."""
    PUSH_NOTIFICATION = "push"
    VOICE_BRIEFING = "voice"
    DESKTOP_POPUP = "desktop"
    MORNING_DIGEST = "morning_digest"
    EVENING_DIGEST = "evening_digest"
    WEEKLY_DIGEST = "weekly_digest"
    EMAIL = "email"
    IN_APP = "in_app"


class EntityType(Enum):
    """Types of entities that can be tracked."""
    PERSON = "person"
    COMPANY = "company"
    BRAND = "brand"
    TOPIC = "topic"
    LOCATION = "location"
    PRODUCT = "product"
    EVENT = "event"
    HASHTAG = "hashtag"
    CUSTOM = "custom"


class RelationshipType(Enum):
    """Relationship types for entities."""
    COMPETITOR = "competitor"
    CLIENT = "client"
    PROSPECT = "prospect"
    PARTNER = "partner"
    INVESTOR = "investor"
    INFLUENCER = "influencer"
    VENDOR = "vendor"
    COLLEAGUE = "colleague"
    TARGET = "target"           # Acquisition target, partnership target
    WATCH = "watch"             # Just monitoring
    SELF = "self"               # User's own brand/account
    OTHER = "other"


class GoalStatus(Enum):
    """Status of a goal."""
    ACTIVE = "active"
    ACHIEVED = "achieved"
    PAUSED = "paused"
    ABANDONED = "abandoned"


class GoalPriority(Enum):
    """Priority levels for goals."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class WatcherStatus(Enum):
    """Status of a watcher."""
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    DISABLED = "disabled"


class UrgencyLevel(Enum):
    """How time-sensitive something is."""
    IMMEDIATE = "immediate"    # Right now
    TODAY = "today"            # Within hours
    THIS_WEEK = "this_week"    # Within days
    WHENEVER = "whenever"      # No time pressure


# =============================================================================
# CONSTANTS
# =============================================================================

# Priority thresholds for delivery decisions
PRIORITY_THRESHOLDS = {
    "critical": 0.9,    # Always deliver immediately
    "high": 0.7,        # Deliver unless user is in focus mode
    "medium": 0.5,      # Queue for next briefing
    "low": 0.3,         # Include in digest only
    "ignore": 0.1,      # Don't surface
}

# Check intervals for different watcher types (in seconds)
CHECK_INTERVALS = {
    "news": 300,        # 5 minutes
    "social": 180,      # 3 minutes
    "calendar": 600,    # 10 minutes
    "market": 60,       # 1 minute
    "screen": 30,       # 30 seconds
    "email": 300,       # 5 minutes
    "default": 300,     # 5 minutes
}

# Relevance score thresholds
RELEVANCE_THRESHOLDS = {
    "entity_match": 0.9,      # Direct entity mention
    "goal_related": 0.8,      # Relates to a goal
    "keyword_match": 0.6,     # Keyword in content
    "source_match": 0.5,      # From tracked source
    "world_general": 0.3,     # Generally in world domain
    "minimum": 0.2,           # Below this, ignore
}

# Signal expiration times (in seconds)
SIGNAL_EXPIRATION = {
    SignalType.NEWS_ARTICLE: 86400,        # 24 hours
    SignalType.SOCIAL_MENTION: 43200,      # 12 hours
    SignalType.SOCIAL_POST: 86400,         # 24 hours
    SignalType.PRICE_CHANGE: 3600,         # 1 hour
    SignalType.CALENDAR_EVENT: 0,          # Never (handled by calendar)
    SignalType.COMPETITOR_ACTIVITY: 86400, # 24 hours
    SignalType.TREND_EMERGING: 21600,      # 6 hours
    SignalType.DEADLINE_APPROACHING: 0,    # Never (until deadline passes)
    "default": 43200,                      # 12 hours
}

# Delivery timing preferences
DELIVERY_WINDOWS = {
    "morning_briefing": {"start": "07:00", "end": "09:00"},
    "evening_summary": {"start": "18:00", "end": "20:00"},
    "business_hours": {"start": "09:00", "end": "18:00"},
    "quiet_hours": {"start": "22:00", "end": "07:00"},
}

# Maximum items per digest
DIGEST_LIMITS = {
    "morning": 10,
    "evening": 5,
    "weekly": 20,
}

# Cross-world connection minimum score
CROSS_WORLD_THRESHOLD = 0.6

# Consciousness stream settings
CONSCIOUSNESS_CONFIG = {
    "max_active_thoughts": 10,
    "max_focus_items": 3,
    "thought_retention_hours": 24,
}

# Storage paths (relative to ~/.aria/)
STORAGE_PATHS = {
    "worlds": "ambient/worlds",
    "signals": "ambient/signals",
    "insights": "ambient/insights",
    "history": "ambient/history",
    "consciousness": "ambient/consciousness",
}

# LLM settings for insight generation
LLM_CONFIG = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens_insight": 500,
    "max_tokens_draft": 1000,
    "temperature_insight": 0.3,
    "temperature_creative": 0.7,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_signal_expiration(signal_type: SignalType) -> int:
    """Get expiration time for a signal type in seconds."""
    return SIGNAL_EXPIRATION.get(signal_type, SIGNAL_EXPIRATION["default"])


def get_check_interval(watcher_name: str) -> int:
    """Get check interval for a watcher type in seconds."""
    return CHECK_INTERVALS.get(watcher_name, CHECK_INTERVALS["default"])


def priority_to_numeric(priority: str) -> float:
    """Convert priority string to numeric value."""
    mapping = {
        "critical": 1.0,
        "high": 0.75,
        "medium": 0.5,
        "low": 0.25,
    }
    return mapping.get(priority.lower(), 0.5)
