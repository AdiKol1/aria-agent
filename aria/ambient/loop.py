"""
Aria Ambient Intelligence - Main Loop

The central orchestration loop that connects all ambient components:
Watchers → Cortex → Actors → Delivery
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from .models import (
    Signal, Insight, World, ConsciousnessState, Thought,
    generate_id, now_iso
)
from .constants import (
    InsightStatus,
    InsightPriority,
    CONSCIOUSNESS_CONFIG,
)
from .world_manager import WorldManager
from .storage import SignalCache, InsightHistory, ConsciousnessStorage, WorldStorage
from .relevance import RelevanceScorer
from .watchers import Watcher, WatcherScheduler
from .cortex import PriorityCalculator, InsightGenerator, ConnectionDetector

logger = logging.getLogger(__name__)


class AmbientLoop:
    """
    The main ambient intelligence loop.

    Orchestrates the continuous observe → think → prepare → deliver cycle.

    Usage:
        loop = AmbientLoop()
        loop.add_watcher(NewsWatcher())
        loop.add_watcher(CalendarWatcher())

        await loop.start()
        # ... runs continuously ...
        await loop.stop()
    """

    def __init__(
        self,
        world_manager: WorldManager = None,
        llm_client: Any = None,
        cycle_interval: int = 60,
        insight_callback: Callable[[Insight], None] = None,
    ):
        """
        Initialize the ambient loop.

        Args:
            world_manager: WorldManager instance. Creates new one if not provided.
            llm_client: Optional Anthropic client for LLM-powered insights
            cycle_interval: Seconds between processing cycles
            insight_callback: Callback function when new insights are generated
        """
        # Core components
        self._world_storage = WorldStorage()
        self._world_manager = world_manager or WorldManager(self._world_storage)
        self._signal_cache = SignalCache()
        self._insight_history = InsightHistory()
        self._consciousness_storage = ConsciousnessStorage()

        # Processing components
        self._relevance_scorer = RelevanceScorer()
        self._priority_calculator = PriorityCalculator()
        self._insight_generator = InsightGenerator(
            llm_client=llm_client,
            priority_calculator=self._priority_calculator
        )
        self._connection_detector = ConnectionDetector()

        # Watcher management
        self._scheduler = WatcherScheduler(
            signal_callback=self._on_signals_received
        )

        # State
        self._consciousness = ConsciousnessState()
        self._cycle_interval = cycle_interval
        self._insight_callback = insight_callback
        self._running = False
        self._started_at: Optional[str] = None
        self._cycle_count = 0
        self._total_insights_generated = 0

        # Load existing consciousness state
        saved_state = self._consciousness_storage.load_state()
        if saved_state:
            self._consciousness = saved_state

    # =========================================================================
    # WATCHER MANAGEMENT
    # =========================================================================

    def add_watcher(self, watcher: Watcher) -> None:
        """Add a watcher to the loop."""
        self._scheduler.register_watcher(watcher)

    def remove_watcher(self, name: str) -> bool:
        """Remove a watcher by name."""
        return self._scheduler.unregister_watcher(name)

    def get_watcher(self, name: str) -> Optional[Watcher]:
        """Get a watcher by name."""
        return self._scheduler.get_watcher(name)

    def list_watchers(self) -> List[str]:
        """List all watcher names."""
        return self._scheduler.list_watchers()

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def start(self) -> None:
        """
        Start the ambient loop.

        Begins:
        1. Running all watchers at their intervals
        2. Processing signals into insights on each cycle
        """
        if self._running:
            logger.warning("Ambient loop already running")
            return

        self._running = True
        self._started_at = now_iso()
        logger.info("Starting ambient intelligence loop")

        # Start watcher scheduler
        await self._scheduler.start()

        # Start main processing loop
        asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the ambient loop."""
        if not self._running:
            return

        self._running = False
        logger.info("Stopping ambient intelligence loop")

        # Stop watchers
        await self._scheduler.stop()

        # Save consciousness state
        self._consciousness.updated_at = now_iso()
        self._consciousness_storage.save_state(self._consciousness)

        logger.info(f"Ambient loop stopped. Cycles: {self._cycle_count}, Insights: {self._total_insights_generated}")

    async def _run_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                await self.run_cycle()
                await asyncio.sleep(self._cycle_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ambient loop cycle: {e}")
                await asyncio.sleep(self._cycle_interval)

    # =========================================================================
    # MAIN CYCLE
    # =========================================================================

    async def run_cycle(self) -> List[Insight]:
        """
        Run a single processing cycle.

        Steps:
        1. Get unprocessed signals from cache
        2. Score signals against all worlds
        3. Generate insights for relevant signals
        4. Detect cross-world connections
        5. Update consciousness state

        Returns:
            List of newly generated insights
        """
        self._cycle_count += 1
        cycle_start = datetime.now()
        new_insights = []

        # 1. Get unprocessed signals
        signals = self._signal_cache.get_unprocessed()
        if not signals:
            return new_insights

        logger.debug(f"Processing {len(signals)} signals")

        # 2. Get active worlds
        worlds = self._world_manager.get_active_worlds()
        if not worlds:
            logger.debug("No active worlds")
            return new_insights

        # 3. Score and process each signal
        for signal in signals:
            try:
                insight = await self._process_signal(signal, worlds)
                if insight:
                    new_insights.append(insight)
            except Exception as e:
                logger.error(f"Error processing signal {signal.id}: {e}")

        # 4. Update consciousness
        self._update_consciousness(new_insights)

        # 5. Save state periodically
        if self._cycle_count % 5 == 0:
            self._consciousness_storage.save_state(self._consciousness)

        # Log cycle stats
        cycle_time = (datetime.now() - cycle_start).total_seconds()
        logger.debug(f"Cycle {self._cycle_count}: {len(new_insights)} insights in {cycle_time:.2f}s")

        return new_insights

    async def _process_signal(
        self,
        signal: Signal,
        worlds: List[World]
    ) -> Optional[Insight]:
        """
        Process a single signal.

        Returns:
            Generated Insight, or None if signal not relevant enough
        """
        # Score against all worlds
        world_matches = self._relevance_scorer.score_signal_all_worlds(signal, worlds)

        if not world_matches:
            return None

        # Use best match
        best_match = world_matches[0]
        best_world = self._world_manager.get_world(best_match.world_id)

        if not best_world:
            return None

        # Update signal with matches
        signal.relevant_worlds = world_matches

        # Generate insight
        insight = await self._insight_generator.generate_insight(
            signal=signal,
            world=best_world,
            world_match=best_match,
            context=self._get_cycle_context()
        )

        # Check for cross-world connections
        if len(world_matches) > 1:
            connections = self._connection_detector.find_connections(
                signal, world_matches
            )
            insight.cross_world_connections = connections

        # Record insight
        self._insight_history.record_insight(insight)
        self._total_insights_generated += 1

        # Callback
        if self._insight_callback:
            try:
                self._insight_callback(insight)
            except Exception as e:
                logger.error(f"Insight callback error: {e}")

        return insight

    def _get_cycle_context(self) -> Dict[str, Any]:
        """Get context for insight generation."""
        return {
            "cycle_count": self._cycle_count,
            "active_focus": self._consciousness.current_focus,
            "recent_thoughts": [t.content for t in self._consciousness.active_thoughts[-3:]],
        }

    # =========================================================================
    # CONSCIOUSNESS
    # =========================================================================

    def _update_consciousness(self, new_insights: List[Insight]) -> None:
        """Update Aria's consciousness state with new insights."""
        # Update focus based on active worlds
        active_worlds = self._world_manager.get_active_worlds()
        self._consciousness.current_focus = [w.id for w in active_worlds[:3]]

        # Add pending insights
        for insight in new_insights:
            if insight.id not in self._consciousness.pending_insights:
                self._consciousness.pending_insights.append(insight.id)

                # Add thought about the insight
                thought = Thought(
                    id=generate_id("thought"),
                    content=f"New insight: {insight.title}",
                    category="observation",
                    world_id=insight.world_id,
                    related_insight_id=insight.id,
                    priority=insight.priority_score,
                    timestamp=now_iso(),
                )
                self._consciousness.active_thoughts.append(thought)

        # Prune old thoughts
        max_thoughts = CONSCIOUSNESS_CONFIG.get("max_active_thoughts", 10)
        if len(self._consciousness.active_thoughts) > max_thoughts:
            self._consciousness.active_thoughts = self._consciousness.active_thoughts[-max_thoughts:]

        # Prune old pending insights
        if len(self._consciousness.pending_insights) > 20:
            self._consciousness.pending_insights = self._consciousness.pending_insights[-20:]

        self._consciousness.updated_at = now_iso()

    def _on_signals_received(self, signals: List[Signal]) -> None:
        """Callback when watchers produce new signals."""
        self._signal_cache.add_signals(signals)

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get_briefing(self) -> str:
        """
        Get a briefing of current consciousness state.

        Returns:
            Human-readable briefing string
        """
        parts = []

        # Current focus
        if self._consciousness.current_focus:
            focus_worlds = []
            for world_id in self._consciousness.current_focus[:3]:
                world = self._world_manager.get_world(world_id)
                if world:
                    focus_worlds.append(world.name)
            if focus_worlds:
                parts.append(f"Currently focused on: {', '.join(focus_worlds)}")

        # Pending insights
        pending_count = len(self._consciousness.pending_insights)
        if pending_count:
            parts.append(f"You have {pending_count} pending insight(s) to review")

        # Recent thoughts
        recent_thoughts = self._consciousness.active_thoughts[-3:]
        if recent_thoughts:
            thought_summary = [t.content for t in recent_thoughts]
            parts.append(f"Recent observations:\n- " + "\n- ".join(thought_summary))

        if not parts:
            parts.append("No active observations at the moment")

        return "\n\n".join(parts)

    def get_pending_insights(
        self,
        limit: int = 10,
        priority: InsightPriority = None
    ) -> List[Insight]:
        """Get pending insights from recent history."""
        insights = self._insight_history.get_recent(days=7)

        # Filter to pending/new status
        pending = [
            i for i in insights
            if i.status in [InsightStatus.NEW, InsightStatus.READY]
        ]

        # Filter by priority
        if priority:
            pending = [i for i in pending if i.priority == priority]

        # Sort by priority score
        pending.sort(key=lambda i: i.priority_score, reverse=True)

        return pending[:limit]

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the ambient loop."""
        return {
            "running": self._running,
            "started_at": self._started_at,
            "cycle_count": self._cycle_count,
            "cycle_interval": self._cycle_interval,
            "total_insights_generated": self._total_insights_generated,
            "signals_in_cache": self._signal_cache.count(),
            "pending_insights": len(self._consciousness.pending_insights),
            "active_thoughts": len(self._consciousness.active_thoughts),
            "worlds": {
                "total": len(self._world_manager.list_worlds()),
                "active": len(self._world_manager.get_active_worlds()),
            },
            "watchers": self._scheduler.get_status(),
        }

    @property
    def is_running(self) -> bool:
        """Check if loop is running."""
        return self._running

    @property
    def world_manager(self) -> WorldManager:
        """Get the world manager."""
        return self._world_manager

    @property
    def consciousness(self) -> ConsciousnessState:
        """Get current consciousness state."""
        return self._consciousness
