"""
Aria Ambient Intelligence - Digest Compiler

Compiles periodic digests from batched insights and actions.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..models import Insight, PreparedAction, World
from ..constants import (
    InsightPriority,
    DeliveryChannel,
    DIGEST_LIMITS,
)

logger = logging.getLogger(__name__)


class DigestCompiler:
    """
    Compiles insights and actions into periodic digests.

    Digest Types:
    - Morning: Start-of-day briefing with priorities
    - Evening: End-of-day summary and tomorrow's prep
    - Weekly: Week in review with patterns and achievements

    Formats:
    - Text: For display or reading
    - Voice: Conversational format for voice briefing
    - Structured: Dict format for custom rendering
    """

    def __init__(self, world_names: Dict[str, str] = None):
        """
        Initialize the digest compiler.

        Args:
            world_names: Optional mapping of world_id -> world_name
        """
        self._world_names = world_names or {}

    def set_world_names(self, world_names: Dict[str, str]) -> None:
        """Update the world names mapping."""
        self._world_names = world_names

    # =========================================================================
    # MORNING DIGEST
    # =========================================================================

    def compile_morning(
        self,
        insights: List[Insight],
        actions: List[PreparedAction] = None,
        format: str = "text"
    ) -> str:
        """
        Compile morning digest.

        Content:
        - Priority items for today
        - Upcoming deadlines
        - Key opportunities
        - Quick wins

        Args:
            insights: Insights to include
            actions: Optional prepared actions to include
            format: "text", "voice", or "structured"

        Returns:
            Compiled digest string
        """
        limit = DIGEST_LIMITS.get("morning", 10)
        insights = self._prioritize_insights(insights)[:limit]

        if format == "voice":
            return self._compile_morning_voice(insights, actions)
        elif format == "structured":
            return self._compile_morning_structured(insights, actions)
        else:
            return self._compile_morning_text(insights, actions)

    def _compile_morning_text(
        self,
        insights: List[Insight],
        actions: List[PreparedAction] = None
    ) -> str:
        """Compile morning digest as text."""
        lines = []
        today = datetime.now().strftime("%A, %B %d")

        lines.append(f"Good morning! Here's your briefing for {today}.")
        lines.append("")

        # Priority items
        critical = [i for i in insights if i.priority == InsightPriority.CRITICAL]
        high = [i for i in insights if i.priority == InsightPriority.HIGH]

        if critical:
            lines.append("URGENT:")
            for insight in critical:
                lines.append(f"  ! {insight.title}")
                if insight.suggested_action:
                    lines.append(f"    Action: {insight.suggested_action}")
            lines.append("")

        if high:
            lines.append("HIGH PRIORITY:")
            for insight in high:
                lines.append(f"  * {insight.title}")
            lines.append("")

        # Grouped by world
        by_world = self._group_by_world(insights)
        for world_id, world_insights in by_world.items():
            world_name = self._world_names.get(world_id, world_id)
            remaining = [i for i in world_insights
                        if i.priority not in [InsightPriority.CRITICAL, InsightPriority.HIGH]]
            if remaining:
                lines.append(f"{world_name.upper()}:")
                for insight in remaining[:3]:
                    lines.append(f"  - {insight.title}")
                lines.append("")

        # Actions if provided
        if actions:
            lines.append("PREPARED FOR YOU:")
            for action in actions[:3]:
                lines.append(f"  > {action.content[:50]}...")
            lines.append("")

        lines.append("Have a productive day!")

        return "\n".join(lines)

    def _compile_morning_voice(
        self,
        insights: List[Insight],
        actions: List[PreparedAction] = None
    ) -> str:
        """Compile morning digest for voice delivery."""
        parts = []
        today = datetime.now().strftime("%A")

        parts.append(f"Good morning! Here's your briefing for {today}.")

        # Count items
        critical_count = len([i for i in insights if i.priority == InsightPriority.CRITICAL])
        high_count = len([i for i in insights if i.priority == InsightPriority.HIGH])
        total = len(insights)

        if critical_count:
            parts.append(f"You have {critical_count} urgent item{'s' if critical_count > 1 else ''} that need immediate attention.")

        if high_count:
            parts.append(f"There are {high_count} high priority item{'s' if high_count > 1 else ''} to address today.")

        # Top 3 items
        top_insights = self._prioritize_insights(insights)[:3]
        if top_insights:
            parts.append("Here are the top items:")
            for i, insight in enumerate(top_insights, 1):
                parts.append(f"Number {i}: {insight.title}.")
                if insight.suggested_action:
                    parts.append(f"You might want to {insight.suggested_action.lower()}.")

        # Summary
        if total > 3:
            parts.append(f"Plus {total - 3} more items you can review in the app.")

        parts.append("Would you like me to go into detail on any of these?")

        return " ".join(parts)

    def _compile_morning_structured(
        self,
        insights: List[Insight],
        actions: List[PreparedAction] = None
    ) -> Dict[str, Any]:
        """Compile morning digest as structured data."""
        return {
            "type": "morning_digest",
            "date": datetime.now().isoformat(),
            "summary": {
                "total_items": len(insights),
                "critical": len([i for i in insights if i.priority == InsightPriority.CRITICAL]),
                "high": len([i for i in insights if i.priority == InsightPriority.HIGH]),
            },
            "insights": [
                {
                    "id": i.id,
                    "title": i.title,
                    "priority": i.priority.value,
                    "world_id": i.world_id,
                    "action": i.suggested_action,
                }
                for i in insights
            ],
            "actions": [
                {
                    "id": a.id,
                    "type": a.type.value,
                    "preview": a.content[:100],
                }
                for a in (actions or [])
            ],
        }

    # =========================================================================
    # EVENING DIGEST
    # =========================================================================

    def compile_evening(
        self,
        insights: List[Insight],
        actions: List[PreparedAction] = None,
        format: str = "text"
    ) -> str:
        """
        Compile evening digest.

        Content:
        - Day summary
        - Deferred items
        - Tomorrow preview
        """
        limit = DIGEST_LIMITS.get("evening", 5)
        insights = self._prioritize_insights(insights)[:limit]

        if format == "voice":
            return self._compile_evening_voice(insights, actions)
        else:
            return self._compile_evening_text(insights, actions)

    def _compile_evening_text(
        self,
        insights: List[Insight],
        actions: List[PreparedAction] = None
    ) -> str:
        """Compile evening digest as text."""
        lines = []

        lines.append("Good evening! Here's your end-of-day summary.")
        lines.append("")

        # Remaining items
        if insights:
            lines.append("ITEMS TO REVIEW:")
            for insight in insights:
                status = "!" if insight.priority in [InsightPriority.CRITICAL, InsightPriority.HIGH] else "-"
                lines.append(f"  {status} {insight.title}")
            lines.append("")

        # Prepared actions
        if actions:
            lines.append("DRAFTS READY:")
            for action in actions[:3]:
                lines.append(f"  > {action.content[:40]}...")
            lines.append("")

        lines.append("Rest well!")

        return "\n".join(lines)

    def _compile_evening_voice(
        self,
        insights: List[Insight],
        actions: List[PreparedAction] = None
    ) -> str:
        """Compile evening digest for voice."""
        parts = []

        parts.append("Good evening! Here's a quick wrap-up of your day.")

        if insights:
            parts.append(f"You have {len(insights)} item{'s' if len(insights) > 1 else ''} that rolled over to review tomorrow.")

        if actions:
            parts.append(f"I've prepared {len(actions)} draft{'s' if len(actions) > 1 else ''} for your review.")

        parts.append("Have a good evening!")

        return " ".join(parts)

    # =========================================================================
    # WEEKLY DIGEST
    # =========================================================================

    def compile_weekly(
        self,
        insights: List[Insight],
        format: str = "text"
    ) -> str:
        """
        Compile weekly digest.

        Content:
        - Week highlights
        - Patterns detected
        - Achievement summary
        - Next week preview
        """
        limit = DIGEST_LIMITS.get("weekly", 20)
        insights = insights[:limit]

        lines = []
        week_start = (datetime.now() - timedelta(days=7)).strftime("%B %d")
        week_end = datetime.now().strftime("%B %d")

        lines.append(f"WEEKLY SUMMARY: {week_start} - {week_end}")
        lines.append("=" * 40)
        lines.append("")

        # Stats
        by_world = self._group_by_world(insights)
        by_priority = self._group_by_priority(insights)

        lines.append("OVERVIEW:")
        lines.append(f"  Total insights: {len(insights)}")
        lines.append(f"  Worlds active: {len(by_world)}")
        lines.append("")

        # By world
        lines.append("BY WORLD:")
        for world_id, world_insights in by_world.items():
            world_name = self._world_names.get(world_id, world_id)
            lines.append(f"  {world_name}: {len(world_insights)} items")
        lines.append("")

        # By priority
        lines.append("BY PRIORITY:")
        for priority, priority_insights in by_priority.items():
            lines.append(f"  {priority}: {len(priority_insights)}")
        lines.append("")

        # Highlights
        top_insights = self._prioritize_insights(insights)[:5]
        if top_insights:
            lines.append("TOP INSIGHTS THIS WEEK:")
            for insight in top_insights:
                lines.append(f"  * {insight.title}")
            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _prioritize_insights(self, insights: List[Insight]) -> List[Insight]:
        """Sort insights by priority and score."""
        priority_order = {
            InsightPriority.CRITICAL: 0,
            InsightPriority.HIGH: 1,
            InsightPriority.MEDIUM: 2,
            InsightPriority.LOW: 3,
        }

        return sorted(
            insights,
            key=lambda i: (priority_order.get(i.priority, 2), -i.priority_score)
        )

    def _group_by_world(self, insights: List[Insight]) -> Dict[str, List[Insight]]:
        """Group insights by world."""
        groups = {}
        for insight in insights:
            world_id = insight.world_id or "general"
            if world_id not in groups:
                groups[world_id] = []
            groups[world_id].append(insight)
        return groups

    def _group_by_priority(self, insights: List[Insight]) -> Dict[str, List[Insight]]:
        """Group insights by priority."""
        groups = {}
        for insight in insights:
            priority = insight.priority.value
            if priority not in groups:
                groups[priority] = []
            groups[priority].append(insight)
        return groups

    def get_digest_time(self, digest_type: str) -> Optional[str]:
        """Get the scheduled time for a digest type."""
        windows = {
            "morning": DELIVERY_WINDOWS.get("morning_briefing", {}).get("start", "07:00"),
            "evening": DELIVERY_WINDOWS.get("evening_summary", {}).get("start", "18:00"),
        }
        return windows.get(digest_type)

    def is_digest_time(self, digest_type: str, tolerance_minutes: int = 30) -> bool:
        """Check if it's time for a specific digest."""
        scheduled = self.get_digest_time(digest_type)
        if not scheduled:
            return False

        now = datetime.now()
        scheduled_time = datetime.strptime(scheduled, "%H:%M").time()
        scheduled_dt = datetime.combine(now.date(), scheduled_time)

        diff = abs((now - scheduled_dt).total_seconds() / 60)
        return diff <= tolerance_minutes
