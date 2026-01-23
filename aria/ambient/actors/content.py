"""
Aria Ambient Intelligence - Content Drafter Actor

Drafts social media posts, articles, and responses based on insights.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from .base import Actor, ActorConfig
from ..models import Insight, PreparedAction, QuickAction
from ..constants import ActionType, DeliveryChannel, LLM_CONFIG

logger = logging.getLogger(__name__)


class ContentDrafter(Actor):
    """
    Drafts content based on insights.

    Capabilities:
    - Social media posts (Twitter/X, LinkedIn, etc.)
    - Response drafts (replies, comments)
    - Article outlines
    - Email drafts

    Uses LLM for high-quality content generation when available.
    """

    name = "content_drafter"
    description = "Drafts social media posts, responses, and articles"
    handled_action_types = [
        ActionType.DRAFT_CONTENT,
        ActionType.DRAFT_RESPONSE,
    ]

    def __init__(self, config: ActorConfig = None, llm_client: Any = None):
        super().__init__(config, llm_client)

        # Platform-specific constraints
        self._platform_limits = {
            "twitter": 280,
            "linkedin": 3000,
            "instagram": 2200,
            "facebook": 63206,
            "default": 500,
        }

    async def prepare(self, insight: Insight) -> PreparedAction:
        """
        Prepare content draft for an insight.

        Args:
            insight: The insight to create content for

        Returns:
            PreparedAction with draft content and options
        """
        # Determine content type
        if insight.action_type == ActionType.DRAFT_RESPONSE:
            content, options = await self._draft_response(insight)
        else:
            content, options = await self._draft_post(insight)

        # Create quick actions
        quick_actions = self._create_content_quick_actions(insight)

        return self.create_prepared_action(
            insight=insight,
            content=content,
            options=options,
            quick_actions=quick_actions,
            preferred_channel=DeliveryChannel.IN_APP,
        )

    async def _draft_post(self, insight: Insight) -> tuple:
        """
        Draft a social media post.

        Returns:
            Tuple of (main_content, [alternative_options])
        """
        platform = self.get_setting("default_platform", "twitter")
        char_limit = self._platform_limits.get(platform, 500)

        if self._llm:
            return await self._llm_draft_post(insight, platform, char_limit)
        else:
            return self._template_draft_post(insight, char_limit)

    async def _llm_draft_post(
        self,
        insight: Insight,
        platform: str,
        char_limit: int
    ) -> tuple:
        """Draft post using LLM."""
        try:
            prompt = f"""Draft a {platform} post about this insight.

Insight: {insight.title}
Summary: {insight.summary}
Suggested Action: {insight.suggested_action}

Requirements:
- Maximum {char_limit} characters
- Engaging and actionable
- Professional but approachable tone
- Include a call-to-action if appropriate

Generate 3 variations:
1. Direct/informative style
2. Engaging/question style
3. Thought leadership style

Format each as:
OPTION 1: [content]
OPTION 2: [content]
OPTION 3: [content]"""

            response = await asyncio.to_thread(
                self._llm.messages.create,
                model=LLM_CONFIG.get("model", "claude-sonnet-4-20250514"),
                max_tokens=LLM_CONFIG.get("max_tokens_draft", 1000),
                temperature=LLM_CONFIG.get("temperature_creative", 0.7),
                messages=[{"role": "user", "content": prompt}]
            )

            return self._parse_options(response.content[0].text, char_limit)

        except Exception as e:
            logger.error(f"LLM draft failed: {e}")
            return self._template_draft_post(insight, char_limit)

    def _template_draft_post(self, insight: Insight, char_limit: int) -> tuple:
        """Draft post using templates."""
        # Template 1: Direct
        direct = f"{insight.title}\n\n{insight.summary[:char_limit - len(insight.title) - 20]}"

        # Template 2: Question hook
        question = f"Did you know? {insight.summary[:char_limit - 15]}"

        # Template 3: Action-focused
        action = insight.suggested_action or insight.summary
        action_post = f"Action item: {action[:char_limit - 15]}"

        # Truncate all to limit
        options = [
            direct[:char_limit],
            question[:char_limit],
            action_post[:char_limit],
        ]

        return options[0], options[1:]

    async def _draft_response(self, insight: Insight) -> tuple:
        """
        Draft a response to something.

        Returns:
            Tuple of (main_content, [alternative_options])
        """
        if self._llm:
            return await self._llm_draft_response(insight)
        else:
            return self._template_draft_response(insight)

    async def _llm_draft_response(self, insight: Insight) -> tuple:
        """Draft response using LLM."""
        try:
            prompt = f"""Draft a response based on this insight.

Context: {insight.title}
Details: {insight.summary}
Action needed: {insight.suggested_action}

Generate 3 response variations:
1. Brief and professional
2. Friendly and engaging
3. Detailed and thorough

Format each as:
OPTION 1: [response]
OPTION 2: [response]
OPTION 3: [response]"""

            response = await asyncio.to_thread(
                self._llm.messages.create,
                model=LLM_CONFIG.get("model", "claude-sonnet-4-20250514"),
                max_tokens=LLM_CONFIG.get("max_tokens_draft", 1000),
                temperature=LLM_CONFIG.get("temperature_creative", 0.7),
                messages=[{"role": "user", "content": prompt}]
            )

            return self._parse_options(response.content[0].text)

        except Exception as e:
            logger.error(f"LLM response draft failed: {e}")
            return self._template_draft_response(insight)

    def _template_draft_response(self, insight: Insight) -> tuple:
        """Draft response using templates."""
        # Brief
        brief = f"Thanks for sharing. {insight.suggested_action or 'I will look into this.'}"

        # Friendly
        friendly = f"This is interesting! {insight.summary[:100]}. Let me know if you'd like to discuss further."

        # Detailed
        detailed = f"Regarding {insight.title}:\n\n{insight.summary}\n\nNext steps: {insight.suggested_action}"

        return brief, [friendly, detailed]

    def _parse_options(self, text: str, max_length: int = None) -> tuple:
        """Parse LLM response into options."""
        options = []

        for line in text.split("\n"):
            line = line.strip()
            for prefix in ["OPTION 1:", "OPTION 2:", "OPTION 3:"]:
                if line.startswith(prefix):
                    content = line[len(prefix):].strip()
                    if max_length:
                        content = content[:max_length]
                    options.append(content)
                    break

        if not options:
            # Fallback: use whole response
            content = text[:max_length] if max_length else text
            return content, []

        return options[0], options[1:]

    def _create_content_quick_actions(self, insight: Insight) -> List[QuickAction]:
        """Create quick actions for content drafts."""
        actions = []

        # Copy to clipboard
        actions.append(self.create_quick_action(
            label="Copy",
            action_type="copy",
            payload={"target": "clipboard"}
        ))

        # Edit in app
        actions.append(self.create_quick_action(
            label="Edit",
            action_type="edit",
            payload={"insight_id": insight.id}
        ))

        # Schedule for later
        actions.append(self.create_quick_action(
            label="Schedule",
            action_type="schedule",
            payload={"insight_id": insight.id}
        ))

        # Dismiss
        actions.append(self.create_quick_action(
            label="Dismiss",
            action_type="dismiss",
            payload={"insight_id": insight.id}
        ))

        return actions

    async def draft_thread(self, topic: str, points: List[str] = None) -> List[str]:
        """
        Draft a Twitter/X thread.

        Args:
            topic: Main topic of the thread
            points: Key points to cover

        Returns:
            List of thread posts
        """
        if not self._llm:
            # Simple template-based thread
            thread = [f"Thread: {topic}"]
            if points:
                for i, point in enumerate(points[:5], 1):
                    thread.append(f"{i}. {point[:270]}")
            thread.append("That's it! What do you think?")
            return thread

        try:
            prompt = f"""Create a Twitter/X thread about: {topic}

Key points to cover:
{chr(10).join(f'- {p}' for p in (points or [])) or 'Cover the main aspects'}

Requirements:
- 5-7 tweets maximum
- Each tweet max 280 characters
- First tweet should hook the reader
- Last tweet should be a call-to-action
- Use thread numbering (1/, 2/, etc.)

Format each tweet on a new line starting with the number."""

            response = await asyncio.to_thread(
                self._llm.messages.create,
                model=LLM_CONFIG.get("model", "claude-sonnet-4-20250514"),
                max_tokens=1500,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse thread
            thread = []
            for line in response.content[0].text.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("Thread")):
                    thread.append(line[:280])

            return thread if thread else [f"Thread: {topic}"]

        except Exception as e:
            logger.error(f"Thread generation failed: {e}")
            return [f"Thread: {topic}"]
