"""Type Intent Handler - Handles text typing actions.

This module provides the handler for typing text into the active
application or text field.
"""
import re
from typing import Dict, Any, List, Optional
from .base import IntentHandler, Intent, IntentResult, IntentType


class TypeIntentHandler(IntentHandler):
    """Handler for type intents.

    Recognizes and processes commands like:
    - "type hello world"
    - "write my email address"
    - "enter the password"
    - "input some text"
    - "type 'hello there'"
    """

    PATTERNS: List[str] = [
        r'^type\s+["\'](.+?)["\']$',
        r'^type\s+(.+)$',
        r'^write\s+["\'](.+?)["\']$',
        r'^write\s+(.+)$',
        r'^enter\s+["\'](.+?)["\']$',
        r'^enter\s+(.+)$',
        r'^input\s+["\'](.+?)["\']$',
        r'^input\s+(.+)$',
        r'^type\s+in\s+["\'](.+?)["\']$',
        r'^type\s+in\s+(.+)$',
        r'^write\s+out\s+["\'](.+?)["\']$',
        r'^write\s+out\s+(.+)$',
    ]

    # Keywords that strongly indicate a type intent
    KEYWORDS: List[str] = ["type", "write", "enter", "input"]

    # Special text patterns that need handling
    SPECIAL_PATTERNS = {
        "my email": "{user_email}",
        "my email address": "{user_email}",
        "my name": "{user_name}",
        "my phone": "{user_phone}",
        "my phone number": "{user_phone}",
        "my address": "{user_address}",
        "today's date": "{today}",
        "current date": "{today}",
        "current time": "{now}",
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

        # Check for keyword presence at the start
        for keyword in self.KEYWORDS:
            if text_lower.startswith(keyword + " "):
                return 0.95

        return 0.0

    def extract_params(self, text: str) -> Dict[str, Any]:
        """Extract parameters from the input text.

        Args:
            text: The user input text to parse.

        Returns:
            Dictionary containing:
            - text: The text to type
            - has_placeholder: True if text contains placeholders
            - press_enter: True if should press enter after typing
        """
        if not text:
            return {}

        text_lower = text.lower().strip()
        params: Dict[str, Any] = {}

        # Extract the text to type
        text_to_type = self._extract_text(text)
        if text_to_type:
            params["text"] = text_to_type

            # Check for special patterns
            text_lower_content = text_to_type.lower()
            for pattern, placeholder in self.SPECIAL_PATTERNS.items():
                if pattern in text_lower_content:
                    params["has_placeholder"] = True
                    params["placeholder_type"] = placeholder
                    break

        # Check if should press enter after
        if text_lower.endswith(" and enter") or text_lower.endswith(" and press enter"):
            params["press_enter"] = True
            # Remove the suffix from text
            if "text" in params:
                params["text"] = re.sub(r'\s+and\s+(press\s+)?enter$', '', params["text"], flags=re.IGNORECASE)

        # Check if should press tab after
        if text_lower.endswith(" and tab") or text_lower.endswith(" and press tab"):
            params["press_tab"] = True
            if "text" in params:
                params["text"] = re.sub(r'\s+and\s+(press\s+)?tab$', '', params["text"], flags=re.IGNORECASE)

        return params

    def _extract_text(self, text: str) -> Optional[str]:
        """Extract the text to type from the command.

        Args:
            text: The command text.

        Returns:
            The text to type.
        """
        # Try pattern matching, preferring quoted text
        for pattern in self.PATTERNS:
            match = re.match(pattern, text, re.IGNORECASE)
            if match and match.groups():
                return match.group(1)

        # Fallback: remove the action keyword
        for keyword in self.KEYWORDS:
            if text.lower().startswith(keyword + " "):
                return text[len(keyword) + 1:].strip()

        return None

    def handle(self, intent: Intent) -> IntentResult:
        """Execute the type intent.

        Note: This is a placeholder. The actual typing execution would be
        handled by the action executor which interfaces with the MCP tools.

        Args:
            intent: The parsed type intent.

        Returns:
            The result of the type action.
        """
        text_to_type = intent.params.get("text", "")

        if not text_to_type:
            return IntentResult.error_result(
                error="No text specified to type",
                response="I need to know what text to type."
            )

        response_parts = [f"Would type '{text_to_type}'"]

        if intent.params.get("press_enter"):
            response_parts.append("then press Enter")
        if intent.params.get("press_tab"):
            response_parts.append("then press Tab")

        return IntentResult.success_result(
            response=" ".join(response_parts),
            data={
                "action": "type",
                "text": text_to_type,
                "params": intent.params
            }
        )
