"""
Aria Ambient Intelligence - Relevance Scoring

Calculates how relevant signals are to the user's worlds, goals, and entities.
This is the core matching logic that determines which signals matter.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging
import re

from .models import World, Goal, Entity, Signal, WorldMatch
from .constants import (
    RELEVANCE_THRESHOLDS,
    GoalStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class MatchDetail:
    """Details about why something matched."""
    match_type: str  # "entity", "keyword", "goal", "source"
    matched_text: str
    score_contribution: float
    context: str = ""


class RelevanceScorer:
    """
    Scores how relevant signals are to worlds.

    The scorer uses multiple strategies:
    1. Entity matching - Direct mentions of tracked entities
    2. Keyword matching - World keywords in signal content
    3. Goal relevance - Signals related to active goals
    4. Source matching - Signals from tracked information sources

    Scores are normalized to 0.0-1.0 range.
    """

    def __init__(self, fuzzy_matching: bool = False):
        """
        Initialize the scorer.

        Args:
            fuzzy_matching: Enable fuzzy string matching (slower but catches typos)
        """
        self._fuzzy = fuzzy_matching
        self._entity_cache: Dict[str, List[str]] = {}  # world_id -> entity patterns

    def score_signal(self, signal: Signal, world: World) -> WorldMatch:
        """
        Score how relevant a signal is to a world.

        Args:
            signal: Signal to score
            world: World to match against

        Returns:
            WorldMatch with relevance score and match details
        """
        match_details: List[MatchDetail] = []

        # Combine signal text for matching
        signal_text = f"{signal.title} {signal.content}".lower()

        # 1. Entity matching (highest weight)
        entity_matches = self._score_entity_matches(signal_text, world)
        matched_entity_ids = []
        for entity, score, detail in entity_matches:
            match_details.append(detail)
            matched_entity_ids.append(entity.id)

        # 2. Keyword matching
        keyword_matches = self._score_keyword_matches(signal_text, world)
        matched_keywords = []
        for keyword, score, detail in keyword_matches:
            match_details.append(detail)
            matched_keywords.append(keyword)

        # 3. Goal relevance
        goal_matches = self._score_goal_relevance(signal, world)
        matched_goal_ids = []
        for goal, score, detail in goal_matches:
            match_details.append(detail)
            matched_goal_ids.append(goal.id)

        # 4. Source matching
        source_score = self._score_source_match(signal, world)
        if source_score > 0:
            match_details.append(MatchDetail(
                match_type="source",
                matched_text=signal.url or signal.source,
                score_contribution=source_score,
            ))

        # Calculate total score
        total_score = sum(d.score_contribution for d in match_details)

        # Normalize to 0-1 (with diminishing returns)
        # Use logistic-like curve to prevent infinite scores
        normalized_score = min(total_score / (total_score + 1.0), 1.0) if total_score > 0 else 0.0

        # Generate match reason
        match_reason = self._generate_match_reason(match_details)

        return WorldMatch(
            world_id=world.id,
            world_name=world.name,
            relevance_score=normalized_score,
            matched_entities=matched_entity_ids,
            matched_keywords=matched_keywords,
            matched_goals=matched_goal_ids,
            match_reason=match_reason,
        )

    def score_signal_all_worlds(
        self,
        signal: Signal,
        worlds: List[World],
        min_score: float = None
    ) -> List[WorldMatch]:
        """
        Score a signal against all worlds.

        Args:
            signal: Signal to score
            worlds: List of worlds to match against
            min_score: Minimum score to include (defaults to RELEVANCE_THRESHOLDS["minimum"])

        Returns:
            List of WorldMatch objects, sorted by relevance (highest first)
        """
        min_score = min_score or RELEVANCE_THRESHOLDS.get("minimum", 0.2)

        matches = []
        for world in worlds:
            match = self.score_signal(signal, world)
            if match.relevance_score >= min_score:
                matches.append(match)

        # Sort by relevance
        matches.sort(key=lambda m: m.relevance_score, reverse=True)
        return matches

    # =========================================================================
    # ENTITY MATCHING
    # =========================================================================

    def _score_entity_matches(
        self,
        text: str,
        world: World
    ) -> List[Tuple[Entity, float, MatchDetail]]:
        """
        Find entity matches in text.

        Returns:
            List of (entity, score, detail) tuples
        """
        matches = []

        for entity in world.entities:
            if entity.matches_text(text):
                # Score based on entity importance
                score = RELEVANCE_THRESHOLDS["entity_match"] * entity.importance

                detail = MatchDetail(
                    match_type="entity",
                    matched_text=entity.name,
                    score_contribution=score,
                    context=f"Entity '{entity.name}' ({entity.relationship.value})",
                )
                matches.append((entity, score, detail))

        return matches

    def score_entity_match(self, signal: Signal, entity: Entity) -> float:
        """
        Score how well a signal matches a specific entity.

        Args:
            signal: Signal to check
            entity: Entity to match

        Returns:
            Match score (0.0-1.0)
        """
        text = f"{signal.title} {signal.content}".lower()

        if not entity.matches_text(text):
            return 0.0

        base_score = RELEVANCE_THRESHOLDS["entity_match"]

        # Boost for title match (more prominent)
        if entity.matches_text(signal.title.lower()):
            base_score *= 1.2

        # Check for watched events
        if entity.watch_for:
            text_lower = text.lower()
            for event in entity.watch_for:
                if event.lower() in text_lower:
                    base_score *= 1.5
                    break

        return min(base_score * entity.importance, 1.0)

    # =========================================================================
    # KEYWORD MATCHING
    # =========================================================================

    def _score_keyword_matches(
        self,
        text: str,
        world: World
    ) -> List[Tuple[str, float, MatchDetail]]:
        """
        Find keyword matches in text.

        Returns:
            List of (keyword, score, detail) tuples
        """
        matches = []

        for keyword in world.keywords:
            keyword_lower = keyword.lower()

            # Word boundary matching (avoid partial matches)
            pattern = r'\b' + re.escape(keyword_lower) + r'\b'
            if re.search(pattern, text):
                score = RELEVANCE_THRESHOLDS["keyword_match"]

                detail = MatchDetail(
                    match_type="keyword",
                    matched_text=keyword,
                    score_contribution=score,
                )
                matches.append((keyword, score, detail))

        return matches

    def score_keyword_match(self, signal: Signal, keywords: List[str]) -> float:
        """
        Score signal against a list of keywords.

        Args:
            signal: Signal to check
            keywords: Keywords to match

        Returns:
            Total keyword match score
        """
        text = f"{signal.title} {signal.content}".lower()
        total_score = 0.0

        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            matches = re.findall(pattern, text)
            if matches:
                # Diminishing returns for multiple matches
                keyword_score = RELEVANCE_THRESHOLDS["keyword_match"]
                keyword_score *= min(1 + len(matches) * 0.1, 1.5)
                total_score += keyword_score

        return min(total_score, 1.0)

    # =========================================================================
    # GOAL RELEVANCE
    # =========================================================================

    def _score_goal_relevance(
        self,
        signal: Signal,
        world: World
    ) -> List[Tuple[Goal, float, MatchDetail]]:
        """
        Find goals that the signal might be relevant to.

        Returns:
            List of (goal, score, detail) tuples
        """
        matches = []
        text = f"{signal.title} {signal.content}".lower()

        for goal in world.goals:
            if goal.status != GoalStatus.ACTIVE:
                continue

            score = self.score_goal_relevance(signal, goal)
            if score > 0:
                detail = MatchDetail(
                    match_type="goal",
                    matched_text=goal.description[:50],
                    score_contribution=score,
                    context=f"Goal: {goal.description[:100]}",
                )
                matches.append((goal, score, detail))

        return matches

    def score_goal_relevance(self, signal: Signal, goal: Goal) -> float:
        """
        Score how relevant a signal is to a goal.

        Args:
            signal: Signal to check
            goal: Goal to match

        Returns:
            Relevance score (0.0-1.0)
        """
        text = f"{signal.title} {signal.content}".lower()
        score = 0.0

        # Check progress indicators
        for indicator in goal.progress_indicators:
            if indicator.lower() in text:
                score += RELEVANCE_THRESHOLDS["goal_related"] * 0.6

        # Check risk indicators (also relevant, might be negative)
        for indicator in goal.risk_indicators:
            if indicator.lower() in text:
                score += RELEVANCE_THRESHOLDS["goal_related"] * 0.4

        # Check if goal description keywords appear
        goal_words = set(goal.description.lower().split())
        # Filter out common words
        goal_words = {w for w in goal_words if len(w) > 3}
        text_words = set(text.split())

        overlap = goal_words & text_words
        if overlap:
            score += len(overlap) * 0.1

        # Priority boost
        priority_multiplier = {
            "critical": 1.5,
            "high": 1.2,
            "medium": 1.0,
            "low": 0.8,
        }
        score *= priority_multiplier.get(goal.priority.value, 1.0)

        return min(score, 1.0)

    # =========================================================================
    # SOURCE MATCHING
    # =========================================================================

    def _score_source_match(self, signal: Signal, world: World) -> float:
        """
        Score if signal comes from a tracked source.

        Returns:
            Match score (0.0-1.0)
        """
        if not signal.url:
            return 0.0

        signal_url = signal.url.lower()

        for source in world.information_sources:
            source_lower = source.lower()
            # Check if source URL is contained in signal URL or vice versa
            if source_lower in signal_url or signal_url in source_lower:
                return RELEVANCE_THRESHOLDS["source_match"]

        return 0.0

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _generate_match_reason(self, details: List[MatchDetail]) -> str:
        """Generate a human-readable match reason."""
        if not details:
            return "No specific matches"

        reasons = []

        entity_matches = [d for d in details if d.match_type == "entity"]
        if entity_matches:
            names = [d.matched_text for d in entity_matches[:3]]
            reasons.append(f"Mentions: {', '.join(names)}")

        keyword_matches = [d for d in details if d.match_type == "keyword"]
        if keyword_matches:
            keywords = [d.matched_text for d in keyword_matches[:3]]
            reasons.append(f"Keywords: {', '.join(keywords)}")

        goal_matches = [d for d in details if d.match_type == "goal"]
        if goal_matches:
            reasons.append(f"Related to {len(goal_matches)} goal(s)")

        source_matches = [d for d in details if d.match_type == "source"]
        if source_matches:
            reasons.append("From tracked source")

        return "; ".join(reasons) if reasons else "General relevance"

    def clear_cache(self) -> None:
        """Clear the entity pattern cache."""
        self._entity_cache.clear()
