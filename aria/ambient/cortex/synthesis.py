"""
Aria Ambient Intelligence - Insight Generation

Synthesizes signals into actionable insights using LLM.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models import (
    Signal, Insight, World, Goal, WorldMatch,
    Connection, generate_id, now_iso
)
from ..constants import (
    InsightPriority,
    InsightStatus,
    ActionType,
    UrgencyLevel,
    LLM_CONFIG,
)
from .priority import PriorityCalculator

logger = logging.getLogger(__name__)


class InsightGenerator:
    """
    Generates human-readable insights from signals.

    The generator:
    1. Takes scored signals with world matches
    2. Generates concise, actionable insight summaries
    3. Suggests appropriate actions
    4. Detects cross-world connections

    Uses LLM for high-quality summarization when available,
    falls back to template-based generation otherwise.
    """

    def __init__(
        self,
        llm_client: Any = None,
        priority_calculator: PriorityCalculator = None
    ):
        """
        Initialize the insight generator.

        Args:
            llm_client: Optional LLM client (Anthropic SDK). If not provided,
                       uses template-based generation.
            priority_calculator: Optional priority calculator instance
        """
        self._llm = llm_client
        self._priority_calc = priority_calculator or PriorityCalculator()
        self._model = LLM_CONFIG.get("model", "claude-sonnet-4-20250514")
        self._max_tokens = LLM_CONFIG.get("max_tokens_insight", 500)
        self._temperature = LLM_CONFIG.get("temperature_insight", 0.3)

    async def generate_insight(
        self,
        signal: Signal,
        world: World,
        world_match: WorldMatch,
        context: Dict[str, Any] = None
    ) -> Insight:
        """
        Generate an insight from a signal.

        Args:
            signal: The source signal
            world: The matched world
            world_match: The relevance match details
            context: Optional additional context

        Returns:
            Generated Insight object
        """
        # Calculate priority
        priority_score, priority_level, urgency = self._priority_calc.calculate(
            signal, world, world_match.relevance_score
        )

        # Generate insight content
        if self._llm:
            title, summary, suggested_action = await self._llm_generate(
                signal, world, world_match, context
            )
        else:
            title, summary, suggested_action = self._template_generate(
                signal, world, world_match
            )

        # Determine action type
        action_type = self._determine_action_type(signal, world, suggested_action)

        # Create insight
        insight = Insight(
            id=generate_id("insight"),
            signal_ids=[signal.id],
            title=title,
            summary=summary,
            world_id=world.id,
            priority=priority_level,
            priority_score=priority_score,
            urgency=urgency,
            suggested_action=suggested_action,
            action_type=action_type,
            related_goal_ids=world_match.matched_goals,
            related_entity_ids=world_match.matched_entities,
            status=InsightStatus.NEW,
            created_at=now_iso(),
        )

        return insight

    async def generate_batch(
        self,
        signals_with_matches: List[tuple],  # List of (Signal, World, WorldMatch)
        context: Dict[str, Any] = None
    ) -> List[Insight]:
        """
        Generate insights for multiple signals.

        Args:
            signals_with_matches: List of (signal, world, match) tuples
            context: Optional shared context

        Returns:
            List of generated Insights
        """
        insights = []

        for signal, world, match in signals_with_matches:
            try:
                insight = await self.generate_insight(signal, world, match, context)
                insights.append(insight)
            except Exception as e:
                logger.error(f"Error generating insight: {e}")

        return insights

    async def _llm_generate(
        self,
        signal: Signal,
        world: World,
        match: WorldMatch,
        context: Dict[str, Any] = None
    ) -> tuple:
        """
        Generate insight using LLM.

        Returns:
            Tuple of (title, summary, suggested_action)
        """
        try:
            prompt = self._build_prompt(signal, world, match, context)

            response = await asyncio.to_thread(
                self._llm.messages.create,
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.content[0].text
            return self._parse_llm_response(content)

        except Exception as e:
            logger.error(f"LLM generation failed, falling back to template: {e}")
            return self._template_generate(signal, world, match)

    def _build_prompt(
        self,
        signal: Signal,
        world: World,
        match: WorldMatch,
        context: Dict[str, Any] = None
    ) -> str:
        """Build the LLM prompt for insight generation."""
        active_goals = [g.description for g in world.goals if g.status.value == "active"]

        return f"""Generate a brief, actionable insight from this signal for the user.

World Context: {world.name} - {world.description}
Active Goals: {', '.join(active_goals[:3]) if active_goals else 'None specified'}
Match Reason: {match.match_reason}

Signal:
Title: {signal.title}
Content: {signal.content[:500]}
Source: {signal.source}
Type: {signal.type.value}

Generate:
1. TITLE: A concise title (5-10 words) that captures the key insight
2. SUMMARY: A 1-2 sentence summary explaining why this matters to the user
3. ACTION: A specific action the user could take (or "No action needed" if informational only)

Format your response exactly as:
TITLE: [your title]
SUMMARY: [your summary]
ACTION: [your suggested action]"""

    def _parse_llm_response(self, response: str) -> tuple:
        """Parse LLM response into components."""
        title = ""
        summary = ""
        action = ""

        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("TITLE:"):
                title = line[6:].strip()
            elif line.startswith("SUMMARY:"):
                summary = line[8:].strip()
            elif line.startswith("ACTION:"):
                action = line[7:].strip()

        # Fallbacks
        if not title:
            title = response[:50].split("\n")[0]
        if not summary:
            summary = response[:200]

        return title, summary, action

    def _template_generate(
        self,
        signal: Signal,
        world: World,
        match: WorldMatch
    ) -> tuple:
        """
        Generate insight using templates (fallback when no LLM).

        Returns:
            Tuple of (title, summary, suggested_action)
        """
        # Generate title from signal
        title = signal.title[:60] if signal.title else "New Signal"

        # Generate summary based on match
        if match.matched_entities:
            entity_names = match.matched_entities[:2]
            summary = f"Activity related to {', '.join(entity_names)} detected in {world.name}."
        elif match.matched_keywords:
            keywords = match.matched_keywords[:2]
            summary = f"Signal matches keywords [{', '.join(keywords)}] in your {world.name} world."
        elif match.matched_goals:
            summary = f"This may be relevant to your goals in {world.name}."
        else:
            summary = f"New signal detected that may be relevant to {world.name}."

        # Add content snippet
        if signal.content:
            snippet = signal.content[:100].replace("\n", " ")
            summary = f"{summary} {snippet}..."

        # Generate action suggestion
        action = self._suggest_action_template(signal, world, match)

        return title, summary, action

    def _suggest_action_template(
        self,
        signal: Signal,
        world: World,
        match: WorldMatch
    ) -> str:
        """Generate action suggestion using templates."""
        signal_type = signal.type.value

        if signal_type == "calendar_event":
            return "Review event details and prepare if needed"
        elif signal_type == "calendar_reminder":
            return "Take action on the reminder"
        elif signal_type == "social_mention":
            return "Consider responding or engaging"
        elif signal_type == "news_article":
            if match.matched_entities:
                return f"Review for relevant information about {match.matched_entities[0]}"
            return "Review if relevant to your work"
        elif signal_type == "opportunity_detected":
            return "Evaluate the opportunity"
        elif signal_type == "deadline_approaching":
            return "Prioritize completing the task"
        elif signal_type == "email_important":
            return "Review and respond"
        else:
            return "Review for potential action"

    def _determine_action_type(
        self,
        signal: Signal,
        world: World,
        suggested_action: str
    ) -> ActionType:
        """Determine the appropriate action type."""
        signal_type = signal.type.value
        action_lower = suggested_action.lower()

        # Check signal type
        if signal_type == "social_mention":
            return ActionType.DRAFT_RESPONSE
        elif signal_type in ["news_article", "trend_emerging"]:
            return ActionType.RESEARCH_BRIEF
        elif signal_type in ["calendar_event", "calendar_reminder"]:
            return ActionType.SCHEDULE_SUGGESTION
        elif signal_type == "deadline_approaching":
            return ActionType.TASK_SUGGESTION
        elif signal_type == "opportunity_detected":
            return ActionType.DRAFT_CONTENT

        # Check action text
        if any(word in action_lower for word in ["respond", "reply", "engage"]):
            return ActionType.DRAFT_RESPONSE
        elif any(word in action_lower for word in ["draft", "write", "post"]):
            return ActionType.DRAFT_CONTENT
        elif any(word in action_lower for word in ["research", "investigate", "review"]):
            return ActionType.RESEARCH_BRIEF
        elif any(word in action_lower for word in ["schedule", "calendar"]):
            return ActionType.SCHEDULE_SUGGESTION
        elif any(word in action_lower for word in ["alert", "notify"]):
            return ActionType.ALERT
        elif "no action" in action_lower:
            return ActionType.NONE

        return ActionType.ALERT

    async def suggest_action(self, insight: Insight) -> str:
        """
        Generate or refine action suggestion for an insight.

        This can be called to get more specific action suggestions
        after initial insight generation.
        """
        if self._llm:
            try:
                prompt = f"""Given this insight, suggest a specific action the user could take.

Insight: {insight.title}
Summary: {insight.summary}
Current suggestion: {insight.suggested_action}

Provide a more specific, actionable next step in 1-2 sentences."""

                response = await asyncio.to_thread(
                    self._llm.messages.create,
                    model=self._model,
                    max_tokens=100,
                    temperature=self._temperature,
                    messages=[{"role": "user", "content": prompt}]
                )

                return response.content[0].text.strip()

            except Exception as e:
                logger.error(f"Action suggestion failed: {e}")

        return insight.suggested_action


class ConnectionDetector:
    """
    Detects cross-world connections and opportunities.

    Finds signals that are relevant to multiple worlds,
    indicating potential synergies or opportunities.
    """

    def __init__(self, min_connection_score: float = 0.6):
        """
        Initialize the connection detector.

        Args:
            min_connection_score: Minimum combined relevance to consider a connection
        """
        self._min_score = min_connection_score

    def find_connections(
        self,
        signal: Signal,
        world_matches: List[WorldMatch]
    ) -> List[Connection]:
        """
        Find cross-world connections for a signal.

        Args:
            signal: The signal to analyze
            world_matches: All world matches for this signal

        Returns:
            List of Connection objects
        """
        connections = []

        # Need at least 2 world matches to have a connection
        if len(world_matches) < 2:
            return connections

        # Sort by relevance
        sorted_matches = sorted(
            world_matches,
            key=lambda m: m.relevance_score,
            reverse=True
        )

        # Check pairs of high-relevance matches
        for i, match1 in enumerate(sorted_matches[:-1]):
            if match1.relevance_score < self._min_score / 2:
                break  # Stop if primary match isn't strong enough

            for match2 in sorted_matches[i + 1:]:
                combined_score = (match1.relevance_score + match2.relevance_score) / 2

                if combined_score >= self._min_score:
                    connection = self._create_connection(
                        signal, match1, match2, combined_score
                    )
                    connections.append(connection)

        return connections

    def _create_connection(
        self,
        signal: Signal,
        match1: WorldMatch,
        match2: WorldMatch,
        score: float
    ) -> Connection:
        """Create a connection between two world matches."""
        description = (
            f"Signal relevant to both {match1.world_name} and {match2.world_name}"
        )

        # Generate opportunity text
        opportunity = self._generate_opportunity(signal, match1, match2)

        return Connection(
            id=generate_id("conn"),
            world_ids=[match1.world_id, match2.world_id],
            description=description,
            opportunity=opportunity,
            confidence=score,
        )

    def _generate_opportunity(
        self,
        signal: Signal,
        match1: WorldMatch,
        match2: WorldMatch
    ) -> str:
        """Generate opportunity description for a connection."""
        # Find common elements
        common_entities = set(match1.matched_entities) & set(match2.matched_entities)
        common_keywords = set(match1.matched_keywords) & set(match2.matched_keywords)

        if common_entities:
            return f"Consider how {', '.join(common_entities)} might benefit both {match1.world_name} and {match2.world_name}"
        elif common_keywords:
            return f"Topics [{', '.join(common_keywords)}] overlap between your worlds - potential for cross-pollination"
        else:
            return f"This signal bridges {match1.world_name} and {match2.world_name} - look for synergies"

    def score_connection(self, connection: Connection) -> float:
        """Get the score for a connection."""
        return connection.confidence
