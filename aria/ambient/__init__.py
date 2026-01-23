"""
Aria Ambient Intelligence

A system for proactive, continuous intelligence across all domains of life.

The ambient system transforms Aria from a reactive assistant into a second mind
that continuously observes, synthesizes, and prepares actions across the user's
multiple "worlds" (domains of work and life).

Key Components:
- World System: Mental models of the user's different domains
- Watchers: Continuous signal collection from various sources
- Cortex: Synthesis layer that generates insights
- Actors: Action preparation (drafts, responses, alerts)
- Delivery: Right-time, right-channel delivery of insights

Usage:
    from aria.ambient import WorldManager, World, Goal, Entity
    from aria.ambient import Signal, Insight, PreparedAction
    from aria.ambient.constants import SignalType, InsightPriority
"""

from .models import (
    # World models
    World,
    Goal,
    Entity,
    Schedule,

    # Signal models
    Signal,
    WorldMatch,

    # Insight models
    Insight,
    Connection,

    # Action models
    PreparedAction,
    QuickAction,

    # Consciousness models
    Thought,
    ConsciousnessState,

    # Utilities
    generate_id,
    now_iso,
)

from .constants import (
    # Enums
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
    WatcherStatus,

    # Constants
    PRIORITY_THRESHOLDS,
    CHECK_INTERVALS,
    RELEVANCE_THRESHOLDS,
    SIGNAL_EXPIRATION,
    DELIVERY_WINDOWS,
    DIGEST_LIMITS,
    CROSS_WORLD_THRESHOLD,
    CONSCIOUSNESS_CONFIG,
    STORAGE_PATHS,
    LLM_CONFIG,

    # Helper functions
    get_signal_expiration,
    get_check_interval,
    priority_to_numeric,
)

from .world_manager import WorldManager

from .storage import (
    WorldStorage,
    SignalCache,
    InsightHistory,
    ConsciousnessStorage,
    get_storage_path,
)

from .relevance import RelevanceScorer
from .loop import AmbientLoop
from .integration import AmbientSystem, get_ambient_system

__version__ = "3.0.0-alpha"

__all__ = [
    # World models
    "World",
    "Goal",
    "Entity",
    "Schedule",

    # Signal models
    "Signal",
    "WorldMatch",

    # Insight models
    "Insight",
    "Connection",

    # Action models
    "PreparedAction",
    "QuickAction",

    # Consciousness models
    "Thought",
    "ConsciousnessState",

    # Enums
    "SignalType",
    "InsightPriority",
    "InsightStatus",
    "ActionType",
    "DeliveryChannel",
    "EntityType",
    "RelationshipType",
    "GoalStatus",
    "GoalPriority",
    "UrgencyLevel",
    "WatcherStatus",

    # Utilities
    "generate_id",
    "now_iso",

    # World Manager
    "WorldManager",

    # Storage
    "WorldStorage",
    "SignalCache",
    "InsightHistory",
    "ConsciousnessStorage",
    "get_storage_path",

    # Relevance
    "RelevanceScorer",

    # Main Loop
    "AmbientLoop",

    # Integration
    "AmbientSystem",
    "get_ambient_system",
]
