"""Scroll Intent Handler - Handles scrolling actions.

This module provides the handler for scrolling in the active window,
including directional scrolling and scroll-to-position commands.
"""
import re
from typing import Dict, Any, List, Optional
from .base import IntentHandler, Intent, IntentResult, IntentType


class ScrollIntentHandler(IntentHandler):
    """Handler for scroll intents.

    Recognizes and processes commands like:
    - "scroll down"
    - "scroll up a bit"
    - "page down"
    - "scroll to the bottom"
    - "scroll left"
    - "go down 3 pages"
    """

    PATTERNS: List[str] = [
        r'^scroll\s+(up|down|left|right)(?:\s+(.+))?$',
        r'^page\s+(up|down)$',
        r'^scroll\s+to\s+(?:the\s+)?(top|bottom|beginning|end)$',
        r'^go\s+(up|down)(?:\s+(.+))?$',
        r'^move\s+(up|down)(?:\s+(.+))?$',
    ]

    # Keywords that strongly indicate a scroll intent
    KEYWORDS: List[str] = ["scroll", "page up", "page down"]

    # Default scroll amounts (in pixels)
    SCROLL_AMOUNTS = {
        "tiny": 50,
        "small": 100,
        "a bit": 150,
        "a little": 150,
        "some": 200,
        "normal": 300,
        "medium": 300,
        "a lot": 500,
        "much": 500,
        "large": 600,
        "huge": 800,
        "a page": 300,
        "one page": 300,
        "two pages": 600,
        "three pages": 900,
    }

    # Direction mappings
    DIRECTIONS = {
        "up": "up",
        "down": "down",
        "left": "left",
        "right": "right",
        "top": "top",
        "bottom": "bottom",
        "beginning": "top",
        "end": "bottom",
        "start": "top",
    }

    def can_handle(self, text: str) -> float:
        """Determine if this handler can process the given text.

        Args:
            text: The user input text to evaluate.

        Returns:
            Confidence score from 0.0 to 1.0.
        """
        if not text:
            return 0.0

        text_lower = text.lower().strip()

        # Check for exact pattern matches
        for pattern in self.PATTERNS:
            match = re.match(pattern, text_lower, re.IGNORECASE)
            if match:
                return 1.0

        # Check for keyword presence
        for keyword in self.KEYWORDS:
            if keyword in text_lower:
                return 0.9

        # Check for direction words with movement verbs
        if any(d in text_lower for d in ["up", "down", "left", "right"]):
            if any(v in text_lower for v in ["go", "move", "scroll"]):
                return 0.8

        return 0.0

    def extract_params(self, text: str) -> Dict[str, Any]:
        """Extract parameters from the input text.

        Args:
            text: The user input text to parse.

        Returns:
            Dictionary containing:
            - direction: up, down, left, right, top, or bottom
            - amount: Scroll amount in pixels
            - scroll_to: True if scrolling to a position (top/bottom)
        """
        if not text:
            return {}

        text_lower = text.lower().strip()
        params: Dict[str, Any] = {}

        # Extract direction
        direction = self._extract_direction(text_lower)
        if direction:
            params["direction"] = direction

            # Check if scrolling to position
            if direction in ["top", "bottom"]:
                params["scroll_to"] = True

        # Extract amount
        amount = self._extract_amount(text_lower)
        params["amount"] = amount

        # Handle page up/down
        if "page up" in text_lower:
            params["direction"] = "up"
            params["amount"] = 300
        elif "page down" in text_lower:
            params["direction"] = "down"
            params["amount"] = 300

        return params

    def _extract_direction(self, text: str) -> Optional[str]:
        """Extract scroll direction from text.

        Args:
            text: The command text.

        Returns:
            The direction (up, down, left, right, top, bottom).
        """
        # Check each direction
        for keyword, direction in self.DIRECTIONS.items():
            # Look for the direction word
            if re.search(rf'\b{keyword}\b', text):
                return direction

        return None

    def _extract_amount(self, text: str) -> int:
        """Extract scroll amount from text.

        Args:
            text: The command text.

        Returns:
            Scroll amount in pixels.
        """
        # Check for numeric amounts
        numeric_match = re.search(r'(\d+)\s*(?:pixels?|px|lines?)?', text)
        if numeric_match:
            return int(numeric_match.group(1))

        # Check for page counts
        page_match = re.search(r'(\d+)\s*pages?', text)
        if page_match:
            return int(page_match.group(1)) * 300

        # Check for descriptive amounts
        for amount_key, amount_value in self.SCROLL_AMOUNTS.items():
            if amount_key in text:
                return amount_value

        # Default amount
        return 300

    def handle(self, intent: Intent) -> IntentResult:
        """Execute the scroll intent.

        Note: This is a placeholder. The actual scrolling execution would be
        handled by the action executor which interfaces with the MCP tools.

        Args:
            intent: The parsed scroll intent.

        Returns:
            The result of the scroll action.
        """
        direction = intent.params.get("direction", "down")
        amount = intent.params.get("amount", 300)
        scroll_to = intent.params.get("scroll_to", False)

        if scroll_to:
            return IntentResult.success_result(
                response=f"Would scroll to {direction}",
                data={
                    "action": "scroll_to",
                    "position": direction,
                    "params": intent.params
                }
            )

        # Calculate actual scroll value (negative for down/right)
        scroll_value = amount
        if direction in ["down", "right"]:
            scroll_value = -amount

        return IntentResult.success_result(
            response=f"Would scroll {direction} by {amount} pixels",
            data={
                "action": "scroll",
                "direction": direction,
                "amount": scroll_value,
                "params": intent.params
            }
        )
