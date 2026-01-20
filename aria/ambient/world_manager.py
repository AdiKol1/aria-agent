"""
Aria Ambient Intelligence - World Manager

Manages the user's worlds (domains/contexts), including CRUD operations,
context detection, and world activation logic.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from .models import World, Goal, Entity, Schedule, generate_id, now_iso
from .constants import GoalStatus, GoalPriority, EntityType, RelationshipType

logger = logging.getLogger(__name__)


class WorldManager:
    """
    Manages the user's worlds - their different domains of work and life.

    The WorldManager handles:
    - CRUD operations for worlds, goals, and entities
    - Context detection (which worlds are currently active)
    - World learning and suggestions
    """

    def __init__(self, storage: "WorldStorage" = None):
        """
        Initialize the WorldManager.

        Args:
            storage: Storage backend for persisting worlds. If None, uses in-memory storage.
        """
        self._storage = storage
        self._worlds: Dict[str, World] = {}
        self._active_world_ids: List[str] = []

        # Load existing worlds from storage
        if storage:
            self._load_from_storage()

    def _load_from_storage(self) -> None:
        """Load all worlds from storage."""
        if not self._storage:
            return

        world_ids = self._storage.list_world_ids()
        for world_id in world_ids:
            world = self._storage.load_world(world_id)
            if world:
                self._worlds[world_id] = world
                logger.debug(f"Loaded world: {world.name} ({world_id})")

    def _save_world(self, world: World) -> bool:
        """Save a world to storage."""
        if self._storage:
            return self._storage.save_world(world)
        return True

    # =========================================================================
    # WORLD CRUD
    # =========================================================================

    def create_world(
        self,
        name: str,
        description: str,
        keywords: List[str] = None,
        information_sources: List[str] = None,
        schedule: Schedule = None,
    ) -> World:
        """
        Create a new world.

        Args:
            name: Display name for the world (e.g., "Real Estate")
            description: What this world represents
            keywords: Keywords that indicate this world is active
            information_sources: URLs, feeds, etc. to monitor
            schedule: When this world is typically active

        Returns:
            The created World object
        """
        world = World(
            id=generate_id("world"),
            name=name,
            description=description,
            keywords=keywords or [],
            information_sources=information_sources or [],
            schedule=schedule,
            created_at=now_iso(),
            updated_at=now_iso(),
        )

        self._worlds[world.id] = world
        self._save_world(world)

        logger.info(f"Created world: {name} ({world.id})")
        return world

    def get_world(self, world_id: str) -> Optional[World]:
        """Get a world by ID."""
        return self._worlds.get(world_id)

    def get_world_by_name(self, name: str) -> Optional[World]:
        """Get a world by name (case-insensitive)."""
        name_lower = name.lower()
        for world in self._worlds.values():
            if world.name.lower() == name_lower:
                return world
        return None

    def update_world(self, world_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a world's properties.

        Args:
            world_id: ID of the world to update
            updates: Dictionary of field names to new values

        Returns:
            True if successful, False if world not found
        """
        world = self._worlds.get(world_id)
        if not world:
            logger.warning(f"World not found: {world_id}")
            return False

        # Update allowed fields
        allowed_fields = {
            "name", "description", "keywords", "information_sources",
            "schedule", "successful_approaches", "failure_patterns", "confidence"
        }

        for field, value in updates.items():
            if field in allowed_fields:
                if field == "schedule" and isinstance(value, dict):
                    value = Schedule.from_dict(value)
                setattr(world, field, value)

        world.updated_at = now_iso()
        self._save_world(world)

        logger.info(f"Updated world: {world.name}")
        return True

    def delete_world(self, world_id: str) -> bool:
        """
        Delete a world.

        Args:
            world_id: ID of the world to delete

        Returns:
            True if successful, False if world not found
        """
        if world_id not in self._worlds:
            logger.warning(f"World not found for deletion: {world_id}")
            return False

        world = self._worlds.pop(world_id)

        if self._storage:
            self._storage.delete_world(world_id)

        if world_id in self._active_world_ids:
            self._active_world_ids.remove(world_id)

        logger.info(f"Deleted world: {world.name}")
        return True

    def list_worlds(self) -> List[World]:
        """Get all worlds."""
        return list(self._worlds.values())

    # =========================================================================
    # GOAL MANAGEMENT
    # =========================================================================

    def add_goal(
        self,
        world_id: str,
        description: str,
        priority: GoalPriority = GoalPriority.MEDIUM,
        deadline: str = None,
        progress_indicators: List[str] = None,
        risk_indicators: List[str] = None,
    ) -> Optional[Goal]:
        """
        Add a goal to a world.

        Args:
            world_id: ID of the world to add the goal to
            description: What the user wants to achieve
            priority: Goal priority level
            deadline: ISO format deadline (optional)
            progress_indicators: Signals that indicate progress
            risk_indicators: Signals that indicate risk

        Returns:
            The created Goal, or None if world not found
        """
        world = self._worlds.get(world_id)
        if not world:
            logger.warning(f"World not found for goal: {world_id}")
            return None

        goal = Goal(
            id=generate_id("goal"),
            world_id=world_id,
            description=description,
            priority=priority,
            deadline=deadline,
            progress_indicators=progress_indicators or [],
            risk_indicators=risk_indicators or [],
            created_at=now_iso(),
            updated_at=now_iso(),
        )

        world.goals.append(goal)
        world.updated_at = now_iso()
        self._save_world(world)

        logger.info(f"Added goal to {world.name}: {description}")
        return goal

    def update_goal(
        self,
        world_id: str,
        goal_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update a goal within a world."""
        world = self._worlds.get(world_id)
        if not world:
            return False

        for goal in world.goals:
            if goal.id == goal_id:
                allowed_fields = {
                    "description", "priority", "deadline", "progress_indicators",
                    "risk_indicators", "status", "progress", "notes"
                }
                for field, value in updates.items():
                    if field in allowed_fields:
                        if field == "priority":
                            value = GoalPriority(value) if isinstance(value, str) else value
                        elif field == "status":
                            value = GoalStatus(value) if isinstance(value, str) else value
                        setattr(goal, field, value)

                goal.updated_at = now_iso()
                world.updated_at = now_iso()
                self._save_world(world)
                return True

        return False

    def remove_goal(self, world_id: str, goal_id: str) -> bool:
        """Remove a goal from a world."""
        world = self._worlds.get(world_id)
        if not world:
            return False

        original_count = len(world.goals)
        world.goals = [g for g in world.goals if g.id != goal_id]

        if len(world.goals) < original_count:
            world.updated_at = now_iso()
            self._save_world(world)
            return True

        return False

    def get_all_goals(self, status: GoalStatus = None) -> List[Goal]:
        """Get all goals across all worlds, optionally filtered by status."""
        goals = []
        for world in self._worlds.values():
            for goal in world.goals:
                if status is None or goal.status == status:
                    goals.append(goal)
        return goals

    def get_approaching_deadlines(self, days: int = 7) -> List[Goal]:
        """Get goals with deadlines approaching within the given days."""
        approaching = []
        for world in self._worlds.values():
            for goal in world.goals:
                if goal.status == GoalStatus.ACTIVE and goal.is_deadline_approaching(days):
                    approaching.append(goal)
        return approaching

    # =========================================================================
    # ENTITY MANAGEMENT
    # =========================================================================

    def add_entity(
        self,
        world_id: str,
        name: str,
        entity_type: EntityType = EntityType.CUSTOM,
        relationship: RelationshipType = RelationshipType.WATCH,
        importance: float = 0.5,
        watch_for: List[str] = None,
        aliases: List[str] = None,
        urls: List[str] = None,
        notes: str = "",
    ) -> Optional[Entity]:
        """
        Add an entity to track in a world.

        Args:
            world_id: ID of the world
            name: Entity name (e.g., "Compass Real Estate")
            entity_type: Type of entity
            relationship: User's relationship to this entity
            importance: How important to track (0-1)
            watch_for: Events to watch for
            aliases: Alternative names
            urls: URLs to monitor
            notes: Additional context

        Returns:
            The created Entity, or None if world not found
        """
        world = self._worlds.get(world_id)
        if not world:
            logger.warning(f"World not found for entity: {world_id}")
            return None

        entity = Entity(
            id=generate_id("entity"),
            world_id=world_id,
            name=name,
            type=entity_type,
            relationship=relationship,
            importance=importance,
            watch_for=watch_for or [],
            aliases=aliases or [],
            urls=urls or [],
            notes=notes,
            created_at=now_iso(),
            updated_at=now_iso(),
        )

        world.entities.append(entity)
        world.updated_at = now_iso()
        self._save_world(world)

        logger.info(f"Added entity to {world.name}: {name}")
        return entity

    def update_entity(
        self,
        world_id: str,
        entity_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update an entity within a world."""
        world = self._worlds.get(world_id)
        if not world:
            return False

        for entity in world.entities:
            if entity.id == entity_id:
                allowed_fields = {
                    "name", "type", "relationship", "importance",
                    "watch_for", "notes", "aliases", "urls"
                }
                for field, value in updates.items():
                    if field in allowed_fields:
                        if field == "type":
                            value = EntityType(value) if isinstance(value, str) else value
                        elif field == "relationship":
                            value = RelationshipType(value) if isinstance(value, str) else value
                        setattr(entity, field, value)

                entity.updated_at = now_iso()
                world.updated_at = now_iso()
                self._save_world(world)
                return True

        return False

    def remove_entity(self, world_id: str, entity_id: str) -> bool:
        """Remove an entity from a world."""
        world = self._worlds.get(world_id)
        if not world:
            return False

        original_count = len(world.entities)
        world.entities = [e for e in world.entities if e.id != entity_id]

        if len(world.entities) < original_count:
            world.updated_at = now_iso()
            self._save_world(world)
            return True

        return False

    def find_entity_by_name(self, name: str) -> List[tuple]:
        """
        Find entities by name across all worlds.

        Returns:
            List of (world, entity) tuples
        """
        results = []
        name_lower = name.lower()

        for world in self._worlds.values():
            for entity in world.entities:
                if entity.name.lower() == name_lower:
                    results.append((world, entity))
                elif any(alias.lower() == name_lower for alias in entity.aliases):
                    results.append((world, entity))

        return results

    # =========================================================================
    # CONTEXT DETECTION
    # =========================================================================

    def get_active_worlds(self) -> List[World]:
        """
        Get currently active worlds based on schedule and context.

        Returns:
            List of currently active World objects
        """
        active = []
        for world in self._worlds.values():
            if world.is_active_now():
                active.append(world)

        # If no worlds are active by schedule, return all worlds
        if not active:
            active = list(self._worlds.values())

        return active

    def get_world_for_context(
        self,
        context: Dict[str, Any]
    ) -> List[World]:
        """
        Determine which worlds match the given context.

        Args:
            context: Dictionary with keys like:
                - active_app: Current application name
                - active_url: Current browser URL
                - recent_text: Recent text input
                - keywords: Detected keywords

        Returns:
            List of matching worlds, ordered by relevance
        """
        matches = []

        for world in self._worlds.values():
            score = self._score_world_context(world, context)
            if score > 0:
                matches.append((world, score))

        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return [world for world, _ in matches]

    def _score_world_context(
        self,
        world: World,
        context: Dict[str, Any]
    ) -> float:
        """Score how well a world matches the context."""
        score = 0.0

        # Check if schedule is active (base score)
        if world.is_active_now():
            score += 0.2

        # Check keyword matches
        keywords = context.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [keywords]

        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in [k.lower() for k in world.keywords]:
                score += 0.3

        # Check recent text for world keywords
        recent_text = context.get("recent_text", "")
        if recent_text:
            matched_keywords = world.matches_keywords(recent_text)
            score += len(matched_keywords) * 0.2

            # Check for entity mentions
            for entity in world.entities:
                if entity.matches_text(recent_text):
                    score += entity.importance * 0.3

        # Check active URL against information sources
        active_url = context.get("active_url", "")
        if active_url:
            for source in world.information_sources:
                if source in active_url or active_url in source:
                    score += 0.4

        return min(score, 1.0)  # Cap at 1.0

    def is_world_active(self, world_id: str) -> bool:
        """Check if a specific world is currently active."""
        world = self._worlds.get(world_id)
        return world.is_active_now() if world else False

    def set_world_active(self, world_id: str) -> bool:
        """
        Manually set a world as active (user override).

        Returns:
            True if successful
        """
        if world_id not in self._worlds:
            return False

        if world_id not in self._active_world_ids:
            self._active_world_ids.append(world_id)

        world = self._worlds[world_id]
        world.last_active = now_iso()
        self._save_world(world)

        return True

    def set_world_inactive(self, world_id: str) -> bool:
        """Manually set a world as inactive (user override)."""
        if world_id in self._active_world_ids:
            self._active_world_ids.remove(world_id)
            return True
        return False

    # =========================================================================
    # WORLD STATISTICS
    # =========================================================================

    def get_world_stats(self, world_id: str) -> Dict[str, Any]:
        """Get statistics for a world."""
        world = self._worlds.get(world_id)
        if not world:
            return {}

        active_goals = len([g for g in world.goals if g.status == GoalStatus.ACTIVE])
        total_goals = len(world.goals)
        approaching_deadlines = len([
            g for g in world.goals
            if g.status == GoalStatus.ACTIVE and g.is_deadline_approaching()
        ])

        return {
            "id": world.id,
            "name": world.name,
            "is_active": world.is_active_now(),
            "confidence": world.confidence,
            "total_goals": total_goals,
            "active_goals": active_goals,
            "approaching_deadlines": approaching_deadlines,
            "total_entities": len(world.entities),
            "high_importance_entities": len(world.get_high_importance_entities()),
            "keywords_count": len(world.keywords),
            "sources_count": len(world.information_sources),
            "created_at": world.created_at,
            "updated_at": world.updated_at,
            "last_active": world.last_active,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all worlds."""
        return {
            "total_worlds": len(self._worlds),
            "active_worlds": len(self.get_active_worlds()),
            "total_goals": sum(len(w.goals) for w in self._worlds.values()),
            "active_goals": len(self.get_all_goals(GoalStatus.ACTIVE)),
            "approaching_deadlines": len(self.get_approaching_deadlines()),
            "total_entities": sum(len(w.entities) for w in self._worlds.values()),
            "worlds": [self.get_world_stats(w.id) for w in self._worlds.values()],
        }
