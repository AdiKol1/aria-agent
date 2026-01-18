"""Click Intent Handler - Handles mouse click actions.

This module provides the handler for all click-related intents, including
single clicks, double clicks, and right clicks on UI elements.
"""
import re
from typing import Dict, Any, List, Optional
from .base import IntentHandler, Intent, IntentResult, IntentType


class ClickIntentHandler(IntentHandler):
    """Handler for click intents.

    Recognizes and processes commands like:
    - "click on Chrome"
    - "double click the file"
    - "right click on the desktop"
    - "tap the button"
    - "press the submit button"
    - "select the first option"
    """

    PATTERNS: List[str] = [
        r'^click\s+on\s+(?:the\s+)?(.+)$',
        r'^click\s+(?:the\s+)?(.+)$',
        r'^tap\s+(?:on\s+)?(?:the\s+)?(.+)$',
        r'^press\s+(?:on\s+)?(?:the\s+)?(.+)$',
        r'^select\s+(?:the\s+)?(.+)$',
        r'^hit\s+(?:the\s+)?(.+)$',
        r'^double[\s-]?click\s+(?:on\s+)?(?:the\s+)?(.+)$',
        r'^right[\s-]?click\s+(?:on\s+)?(?:the\s+)?(.+)$',
    ]

    # Keywords that strongly indicate a click intent
    KEYWORDS: List[str] = ["click", "tap", "press", "select", "hit"]

    # Modifiers that change the click behavior
    MODIFIERS = {
        "double": {"double_click": True},
        "right": {"right_click": True},
        "middle": {"middle_click": True},
        "triple": {"triple_click": True},
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
                # Double/right click patterns get high confidence
                if "double" in pattern or "right" in pattern:
                    return 0.95
                return 1.0

        # Check for keyword presence
        for keyword in self.KEYWORDS:
            if text_lower.startswith(keyword):
                return 0.8
            if f" {keyword}" in text_lower:
                return 0.5

        return 0.0

    def extract_params(self, text: str) -> Dict[str, Any]:
        """Extract parameters from the input text.

        Args:
            text: The user input text to parse.

        Returns:
            Dictionary containing:
            - target: The element to click on
            - double_click: True if double click
            - right_click: True if right click
            - location: Optional location hint (dock, menu, etc.)
        """
        if not text:
            return {}

        text_lower = text.lower().strip()
        params: Dict[str, Any] = {}

        # Check for click modifiers
        for modifier, modifier_params in self.MODIFIERS.items():
            if modifier in text_lower:
                params.update(modifier_params)

        # Extract target
        target = self._extract_target(text_lower)
        if target:
            params["target"] = target

        # Extract location hints
        location = self._extract_location(text_lower)
        if location:
            params["location"] = location

        # Extract coordinates if specified
        coords = self._extract_coordinates(text_lower)
        if coords:
            params["coordinates"] = coords

        return params

    def _extract_target(self, text: str) -> Optional[str]:
        """Extract the click target from text.

        Args:
            text: The command text.

        Returns:
            The target element name.
        """
        # Try pattern matching first
        for pattern in self.PATTERNS:
            match = re.match(pattern, text, re.IGNORECASE)
            if match and match.groups():
                target = match.group(1)
                return self._clean_target(target)

        return None

    def _clean_target(self, target: str) -> str:
        """Clean up the extracted target string.

        Args:
            target: The raw target string.

        Returns:
            Cleaned target string.
        """
        if not target:
            return ""

        # Remove location suffixes
        location_suffixes = [
            " in the dock", " from the dock", " on the dock",
            " in the menu", " from the menu", " on the menu",
            " in the toolbar", " on the toolbar",
            " in the taskbar", " on the taskbar",
            " please", " for me",
        ]

        target = target.strip()
        for suffix in location_suffixes:
            if target.lower().endswith(suffix):
                target = target[:-len(suffix)].strip()

        # Remove noise words
        noise = ["the ", "a ", "an ", "icon ", "button "]
        for word in noise:
            target = re.sub(f"^{word}", "", target, flags=re.IGNORECASE)

        return target.strip()

    def _extract_location(self, text: str) -> Optional[str]:
        """Extract location hint from text.

        Args:
            text: The command text.

        Returns:
            Location hint (dock, menu, etc.)
        """
        location_patterns = [
            (r'in\s+(?:the\s+)?(\w+)', 1),
            (r'on\s+(?:the\s+)?(\w+)', 1),
            (r'from\s+(?:the\s+)?(\w+)', 1),
        ]

        for pattern, group in location_patterns:
            match = re.search(pattern, text)
            if match:
                location = match.group(group).lower()
                if location in ["dock", "menu", "toolbar", "taskbar", "desktop", "sidebar"]:
                    return location

        return None

    def _extract_coordinates(self, text: str) -> Optional[Dict[str, int]]:
        """Extract x,y coordinates if specified.

        Args:
            text: The command text.

        Returns:
            Dictionary with x and y coordinates.
        """
        # Match patterns like "at 100, 200" or "at position (100, 200)"
        coord_patterns = [
            r'at\s+\(?\s*(\d+)\s*,\s*(\d+)\s*\)?',
            r'position\s+\(?\s*(\d+)\s*,\s*(\d+)\s*\)?',
            r'coordinates?\s+\(?\s*(\d+)\s*,\s*(\d+)\s*\)?',
        ]

        for pattern in coord_patterns:
            match = re.search(pattern, text)
            if match:
                return {
                    "x": int(match.group(1)),
                    "y": int(match.group(2))
                }

        return None

    def handle(self, intent: Intent) -> IntentResult:
        """Execute the click intent.

        Note: This is a placeholder. The actual click execution would be
        handled by the action executor which interfaces with the MCP tools.

        Args:
            intent: The parsed click intent.

        Returns:
            The result of the click action.
        """
        target = intent.target or intent.params.get("target", "element")
        click_type = "click"

        if intent.params.get("double_click"):
            click_type = "double-click"
        elif intent.params.get("right_click"):
            click_type = "right-click"

        # This would normally call the actual click function
        return IntentResult.success_result(
            response=f"Would {click_type} on '{target}'",
            data={
                "action": "click",
                "target": target,
                "params": intent.params
            }
        )
