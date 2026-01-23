"""
Aria Ambient Intelligence - Alert Composer Actor

Composes alert messages for various delivery channels.
"""

import logging
from typing import Any, Dict, List

from .base import Actor, ActorConfig
from ..models import Insight, PreparedAction, QuickAction
from ..constants import ActionType, DeliveryChannel, InsightPriority, UrgencyLevel

logger = logging.getLogger(__name__)


class AlertComposer(Actor):
    """
    Composes alert messages for different delivery channels.

    Handles:
    - Push notifications (concise)
    - Voice briefings (conversational)
    - Desktop popups (scannable)
    - Digest items (summarized)

    Adapts tone and length based on channel and priority.
    """

    name = "alert_composer"
    description = "Composes alerts for various delivery channels"
    handled_action_types = [
        ActionType.ALERT,
        ActionType.TASK_SUGGESTION,
        ActionType.SCHEDULE_SUGGESTION,
        ActionType.QUESTION,
        ActionType.NONE,  # FYI items still need formatting
    ]

    # Character limits by channel
    CHANNEL_LIMITS = {
        DeliveryChannel.PUSH_NOTIFICATION: 150,
        DeliveryChannel.DESKTOP_POPUP: 300,
        DeliveryChannel.VOICE_BRIEFING: 500,
        DeliveryChannel.MORNING_DIGEST: 200,
        DeliveryChannel.EVENING_DIGEST: 200,
        DeliveryChannel.EMAIL: 1000,
        DeliveryChannel.IN_APP: 500,
    }

    async def prepare(self, insight: Insight) -> PreparedAction:
        """
        Prepare alert for an insight.

        Generates content formatted for the appropriate channel.
        """
        # Determine best channel
        channel = self._select_channel(insight)

        # Compose content for the channel
        content = self._compose_for_channel(insight, channel)

        # Generate alternative formats
        options = self._generate_alternatives(insight)

        # Create quick actions
        quick_actions = self._create_alert_quick_actions(insight)

        return self.create_prepared_action(
            insight=insight,
            content=content,
            options=options,
            quick_actions=quick_actions,
            preferred_channel=channel,
        )

    def _select_channel(self, insight: Insight) -> DeliveryChannel:
        """Select the best delivery channel for an insight."""
        # Critical: push notification
        if insight.priority == InsightPriority.CRITICAL:
            return DeliveryChannel.PUSH_NOTIFICATION

        # Immediate urgency: push
        if insight.urgency == UrgencyLevel.IMMEDIATE:
            return DeliveryChannel.PUSH_NOTIFICATION

        # High priority, today urgency: voice briefing
        if insight.priority == InsightPriority.HIGH:
            if insight.urgency == UrgencyLevel.TODAY:
                return DeliveryChannel.VOICE_BRIEFING
            return DeliveryChannel.DESKTOP_POPUP

        # Medium: depends on action type
        if insight.action_type == ActionType.SCHEDULE_SUGGESTION:
            return DeliveryChannel.DESKTOP_POPUP
        elif insight.action_type == ActionType.TASK_SUGGESTION:
            return DeliveryChannel.IN_APP

        # Low priority: digest
        if insight.priority == InsightPriority.LOW:
            return DeliveryChannel.MORNING_DIGEST

        # Default
        return self._config.default_channel

    def _compose_for_channel(
        self,
        insight: Insight,
        channel: DeliveryChannel
    ) -> str:
        """Compose content formatted for a specific channel."""
        limit = self.CHANNEL_LIMITS.get(channel, 300)

        if channel == DeliveryChannel.PUSH_NOTIFICATION:
            return self._compose_push(insight, limit)
        elif channel == DeliveryChannel.VOICE_BRIEFING:
            return self._compose_voice(insight, limit)
        elif channel == DeliveryChannel.DESKTOP_POPUP:
            return self._compose_desktop(insight, limit)
        elif channel in [DeliveryChannel.MORNING_DIGEST, DeliveryChannel.EVENING_DIGEST]:
            return self._compose_digest_item(insight, limit)
        else:
            return self._compose_default(insight, limit)

    def _compose_push(self, insight: Insight, limit: int) -> str:
        """
        Compose push notification.

        Requirements: Very concise, actionable, attention-grabbing.
        """
        # Priority indicator
        priority_emoji = {
            InsightPriority.CRITICAL: "",
            InsightPriority.HIGH: "",
            InsightPriority.MEDIUM: "",
            InsightPriority.LOW: "",
        }
        prefix = priority_emoji.get(insight.priority, "")

        # Core message
        title = insight.title[:50]

        # Action hint if space allows
        action_hint = ""
        if insight.suggested_action and len(title) < limit - 30:
            action_hint = f" - {insight.suggested_action[:limit - len(title) - len(prefix) - 5]}"

        message = f"{prefix} {title}{action_hint}"
        return message[:limit]

    def _compose_voice(self, insight: Insight, limit: int) -> str:
        """
        Compose voice briefing.

        Requirements: Conversational, natural speech, clear structure.
        """
        parts = []

        # Opening based on priority
        if insight.priority == InsightPriority.CRITICAL:
            parts.append("Urgent update:")
        elif insight.priority == InsightPriority.HIGH:
            parts.append("Important:")

        # Main content
        parts.append(insight.title)

        # Summary (conversational)
        if insight.summary:
            summary = insight.summary.replace("\n", " ")
            parts.append(summary)

        # Action suggestion
        if insight.suggested_action:
            parts.append(f"You might want to {insight.suggested_action.lower()}")

        text = " ".join(parts)
        return text[:limit]

    def _compose_desktop(self, insight: Insight, limit: int) -> str:
        """
        Compose desktop popup.

        Requirements: Scannable, structured, includes action.
        """
        lines = []

        # Title
        lines.append(insight.title)

        # Summary (truncated)
        if insight.summary:
            summary = insight.summary[:150].replace("\n", " ")
            lines.append(summary)

        # Suggested action
        if insight.suggested_action:
            lines.append(f"Action: {insight.suggested_action}")

        text = "\n".join(lines)
        return text[:limit]

    def _compose_digest_item(self, insight: Insight, limit: int) -> str:
        """
        Compose digest item.

        Requirements: Brief, scannable, grouped context.
        """
        # Format: [Priority] Title - Brief summary
        priority_marker = {
            InsightPriority.CRITICAL: "[!]",
            InsightPriority.HIGH: "[*]",
            InsightPriority.MEDIUM: "[-]",
            InsightPriority.LOW: "[ ]",
        }
        marker = priority_marker.get(insight.priority, "[-]")

        summary_snippet = ""
        if insight.summary:
            summary_snippet = f" - {insight.summary[:80].replace(chr(10), ' ')}"

        text = f"{marker} {insight.title}{summary_snippet}"
        return text[:limit]

    def _compose_default(self, insight: Insight, limit: int) -> str:
        """Compose default format."""
        parts = [insight.title]

        if insight.summary:
            parts.append(insight.summary)

        if insight.suggested_action:
            parts.append(f"Suggested: {insight.suggested_action}")

        text = "\n\n".join(parts)
        return text[:limit]

    def _generate_alternatives(self, insight: Insight) -> List[str]:
        """Generate alternative content formats."""
        alternatives = []

        # Shorter version
        short = f"{insight.title}: {insight.suggested_action or 'Review when convenient'}"
        alternatives.append(short[:150])

        # More detailed version
        detailed = f"{insight.title}\n\n{insight.summary}\n\nAction: {insight.suggested_action}"
        alternatives.append(detailed[:500])

        return alternatives

    def _create_alert_quick_actions(self, insight: Insight) -> List[QuickAction]:
        """Create quick actions for alerts."""
        actions = []

        # Action type specific
        if insight.action_type == ActionType.SCHEDULE_SUGGESTION:
            actions.append(self.create_quick_action(
                label="Add to Calendar",
                action_type="schedule",
                payload={"insight_id": insight.id}
            ))

        elif insight.action_type == ActionType.TASK_SUGGESTION:
            actions.append(self.create_quick_action(
                label="Add Task",
                action_type="add_task",
                payload={"insight_id": insight.id, "title": insight.title}
            ))

        elif insight.action_type == ActionType.QUESTION:
            actions.append(self.create_quick_action(
                label="Answer",
                action_type="respond",
                payload={"insight_id": insight.id}
            ))

        # Common actions
        actions.append(self.create_quick_action(
            label="Snooze",
            action_type="snooze",
            payload={"insight_id": insight.id, "duration_minutes": 60}
        ))

        actions.append(self.create_quick_action(
            label="Dismiss",
            action_type="dismiss",
            payload={"insight_id": insight.id}
        ))

        return actions

    # =========================================================================
    # CHANNEL-SPECIFIC COMPOSERS
    # =========================================================================

    def compose_push(self, insight: Insight) -> str:
        """Public method to compose push notification."""
        return self._compose_push(insight, self.CHANNEL_LIMITS[DeliveryChannel.PUSH_NOTIFICATION])

    def compose_voice(self, insight: Insight) -> str:
        """Public method to compose voice briefing."""
        return self._compose_voice(insight, self.CHANNEL_LIMITS[DeliveryChannel.VOICE_BRIEFING])

    def compose_digest_item(self, insight: Insight) -> str:
        """Public method to compose digest item."""
        return self._compose_digest_item(insight, self.CHANNEL_LIMITS[DeliveryChannel.MORNING_DIGEST])

    def compose_for_channel(self, insight: Insight, channel: DeliveryChannel) -> str:
        """Public method to compose for any channel."""
        return self._compose_for_channel(insight, channel)
