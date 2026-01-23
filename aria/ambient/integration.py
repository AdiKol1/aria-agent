"""
Aria Ambient Intelligence - Integration Module

Provides a simplified interface for integrating the ambient system
with the rest of Aria (voice commands, MCP server, main loop).
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional
from functools import lru_cache

from .models import World, Goal, Entity, Insight, PreparedAction
from .constants import (
    EntityType, RelationshipType, GoalPriority, GoalStatus,
    InsightPriority, DeliveryChannel,
)
from .world_manager import WorldManager
from .storage import WorldStorage, SignalCache, InsightHistory
from .relevance import RelevanceScorer
from .loop import AmbientLoop
from .watchers import (
    WatcherScheduler, NewsWatcher, CalendarWatcher, ScreenContextWatcher
)
from .cortex import InsightGenerator, PriorityCalculator
from .actors import ContentDrafter, AlertComposer
from .delivery import DeliveryEngine, DigestCompiler

logger = logging.getLogger(__name__)


# Singleton instance
_ambient_system: Optional["AmbientSystem"] = None


def get_ambient_system() -> "AmbientSystem":
    """Get or create the ambient system singleton."""
    global _ambient_system
    if _ambient_system is None:
        _ambient_system = AmbientSystem()
    return _ambient_system


class AmbientSystem:
    """
    High-level interface to the ambient intelligence system.

    Provides simplified methods for:
    - World management
    - Getting briefings and insights
    - Running the ambient loop
    - Handling voice commands related to ambient

    Usage:
        system = get_ambient_system()

        # Create a world
        world = system.create_world("Real Estate", "My real estate business")

        # Add tracking
        system.add_entity(world.id, "Compass", entity_type="company")
        system.add_goal(world.id, "Close 3 deals this quarter")

        # Start monitoring
        await system.start()

        # Get briefing
        briefing = system.get_briefing()
    """

    def __init__(self, llm_client: Any = None):
        """
        Initialize the ambient system.

        Args:
            llm_client: Optional Anthropic client for LLM-powered features
        """
        self._llm = llm_client
        self._loop: Optional[AmbientLoop] = None
        self._initialized = False

        # Core components
        self._storage = WorldStorage()
        self._world_manager = WorldManager(self._storage)
        self._signal_cache = SignalCache()
        self._insight_history = InsightHistory()

        # Processing components
        self._relevance_scorer = RelevanceScorer()
        self._priority_calculator = PriorityCalculator()
        self._insight_generator = InsightGenerator(llm_client=llm_client)

        # Actors
        self._content_drafter = ContentDrafter(llm_client=llm_client)
        self._alert_composer = AlertComposer()

        # Delivery
        self._delivery_engine = DeliveryEngine()
        self._digest_compiler = DigestCompiler()

        # Callbacks
        self._insight_callbacks: List[Callable[[Insight], None]] = []

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def start(self) -> bool:
        """
        Start the ambient system.

        Initializes watchers and begins the observation loop.

        Returns:
            True if started successfully
        """
        if self._loop and self._loop.is_running:
            logger.warning("Ambient system already running")
            return True

        try:
            # Create the main loop
            self._loop = AmbientLoop(
                world_manager=self._world_manager,
                llm_client=self._llm,
                insight_callback=self._on_insight,
            )

            # Add default watchers
            self._loop.add_watcher(ScreenContextWatcher())
            self._loop.add_watcher(CalendarWatcher())

            # News watcher (add feeds if configured)
            news_watcher = NewsWatcher()
            # Could load feed config from storage here
            self._loop.add_watcher(news_watcher)

            # Start the loop
            await self._loop.start()
            self._initialized = True

            logger.info("Ambient system started")
            return True

        except Exception as e:
            logger.error(f"Failed to start ambient system: {e}")
            return False

    async def stop(self) -> None:
        """Stop the ambient system."""
        if self._loop:
            await self._loop.stop()
            logger.info("Ambient system stopped")

    @property
    def is_running(self) -> bool:
        """Check if the system is running."""
        return self._loop is not None and self._loop.is_running

    # =========================================================================
    # WORLD MANAGEMENT
    # =========================================================================

    def create_world(
        self,
        name: str,
        description: str,
        keywords: List[str] = None,
    ) -> World:
        """Create a new world."""
        return self._world_manager.create_world(
            name=name,
            description=description,
            keywords=keywords or [],
        )

    def get_world(self, world_id: str) -> Optional[World]:
        """Get a world by ID."""
        return self._world_manager.get_world(world_id)

    def get_world_by_name(self, name: str) -> Optional[World]:
        """Get a world by name."""
        return self._world_manager.get_world_by_name(name)

    def list_worlds(self) -> List[World]:
        """List all worlds."""
        return self._world_manager.list_worlds()

    def delete_world(self, world_id: str) -> bool:
        """Delete a world."""
        return self._world_manager.delete_world(world_id)

    def add_goal(
        self,
        world_id: str,
        description: str,
        priority: str = "medium",
        deadline: str = None,
    ) -> Optional[Goal]:
        """Add a goal to a world."""
        priority_enum = GoalPriority(priority)
        return self._world_manager.add_goal(
            world_id=world_id,
            description=description,
            priority=priority_enum,
            deadline=deadline,
        )

    def add_entity(
        self,
        world_id: str,
        name: str,
        entity_type: str = "custom",
        relationship: str = "watch",
        importance: float = 0.5,
        watch_for: List[str] = None,
    ) -> Optional[Entity]:
        """Add an entity to track in a world."""
        type_enum = EntityType(entity_type)
        rel_enum = RelationshipType(relationship)
        return self._world_manager.add_entity(
            world_id=world_id,
            name=name,
            entity_type=type_enum,
            relationship=rel_enum,
            importance=importance,
            watch_for=watch_for or [],
        )

    def add_keyword(self, world_id: str, keyword: str) -> bool:
        """Add a keyword to a world."""
        world = self._world_manager.get_world(world_id)
        if not world:
            return False
        if keyword not in world.keywords:
            world.keywords.append(keyword)
            return self._world_manager.update_world(
                world_id, {"keywords": world.keywords}
            )
        return True

    # =========================================================================
    # BRIEFINGS & INSIGHTS
    # =========================================================================

    def get_briefing(self, format: str = "voice") -> str:
        """
        Get a briefing of the current state.

        Args:
            format: "voice" for conversational, "text" for display

        Returns:
            Briefing string
        """
        if self._loop:
            if format == "voice":
                return self._loop.get_briefing()
            else:
                return self._get_text_briefing()
        else:
            return self._get_offline_briefing(format)

    def _get_text_briefing(self) -> str:
        """Get text-format briefing."""
        lines = []

        # World summary
        worlds = self._world_manager.list_worlds()
        lines.append(f"Worlds: {len(worlds)}")
        for world in worlds[:3]:
            status = "active" if world.is_active_now() else "inactive"
            lines.append(f"  - {world.name} ({status})")

        # Pending insights
        if self._loop:
            pending = self._loop.get_pending_insights(limit=5)
            if pending:
                lines.append(f"\nPending insights: {len(pending)}")
                for insight in pending:
                    lines.append(f"  [{insight.priority.value}] {insight.title}")

        return "\n".join(lines)

    def _get_offline_briefing(self, format: str) -> str:
        """Get briefing when loop isn't running."""
        worlds = self._world_manager.list_worlds()

        if not worlds:
            return "No worlds configured yet. Create a world to start ambient monitoring."

        if format == "voice":
            world_names = [w.name for w in worlds[:3]]
            return f"Ambient system is offline. You have {len(worlds)} world{'s' if len(worlds) > 1 else ''} configured: {', '.join(world_names)}. Say 'start ambient' to begin monitoring."
        else:
            return f"Ambient system offline. {len(worlds)} worlds configured."

    def get_pending_insights(
        self,
        limit: int = 10,
        priority: str = None
    ) -> List[Insight]:
        """Get pending insights."""
        if not self._loop:
            return []

        priority_enum = InsightPriority(priority) if priority else None
        return self._loop.get_pending_insights(limit=limit, priority=priority_enum)

    def get_morning_digest(self) -> str:
        """Get the morning digest."""
        insights = self._insight_history.get_recent(days=1)
        world_names = {w.id: w.name for w in self._world_manager.list_worlds()}
        self._digest_compiler.set_world_names(world_names)
        return self._digest_compiler.compile_morning(insights, format="voice")

    def get_evening_digest(self) -> str:
        """Get the evening digest."""
        insights = self._insight_history.get_recent(days=1)
        world_names = {w.id: w.name for w in self._world_manager.list_worlds()}
        self._digest_compiler.set_world_names(world_names)
        return self._digest_compiler.compile_evening(insights, format="voice")

    # =========================================================================
    # NEWS FEED MANAGEMENT
    # =========================================================================

    def add_news_feed(self, url: str) -> bool:
        """Add an RSS feed for news monitoring."""
        if self._loop:
            watcher = self._loop.get_watcher("news")
            if watcher and isinstance(watcher, NewsWatcher):
                watcher.add_feed(url)
                return True
        return False

    def list_news_feeds(self) -> List[str]:
        """List configured news feeds."""
        if self._loop:
            watcher = self._loop.get_watcher("news")
            if watcher and isinstance(watcher, NewsWatcher):
                return watcher.list_feeds()
        return []

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_insight(self, callback: Callable[[Insight], None]) -> None:
        """Register a callback for new insights."""
        self._insight_callbacks.append(callback)

    def _on_insight(self, insight: Insight) -> None:
        """Internal handler for new insights."""
        for callback in self._insight_callbacks:
            try:
                callback(insight)
            except Exception as e:
                logger.error(f"Insight callback error: {e}")

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "initialized": self._initialized,
            "running": self.is_running,
            "worlds": len(self._world_manager.list_worlds()),
            "active_worlds": len(self._world_manager.get_active_worlds()),
        }

        if self._loop:
            status.update(self._loop.get_status())

        return status

    # =========================================================================
    # VOICE COMMAND HANDLERS
    # =========================================================================

    def handle_voice_command(self, command: str) -> Optional[str]:
        """
        Handle ambient-related voice commands.

        Args:
            command: The voice command (lowercase)

        Returns:
            Response string, or None if command not handled
        """
        command = command.lower().strip()

        # Briefing requests
        if any(phrase in command for phrase in [
            "what's going on", "what is going on", "briefing",
            "what should i know", "update me", "status"
        ]):
            return self.get_briefing()

        # Morning digest
        if any(phrase in command for phrase in [
            "morning briefing", "morning digest", "morning update"
        ]):
            return self.get_morning_digest()

        # Evening digest
        if any(phrase in command for phrase in [
            "evening briefing", "evening digest", "evening summary",
            "end of day"
        ]):
            return self.get_evening_digest()

        # World queries
        if "list worlds" in command or "my worlds" in command:
            worlds = self.list_worlds()
            if worlds:
                names = [w.name for w in worlds]
                return f"You have {len(worlds)} worlds: {', '.join(names)}"
            return "You don't have any worlds configured yet."

        # Start/stop
        if "start ambient" in command or "start monitoring" in command:
            # Would need to be async
            return "Starting ambient monitoring... (use await system.start())"

        if "stop ambient" in command or "stop monitoring" in command:
            return "Stopping ambient monitoring... (use await system.stop())"

        return None
