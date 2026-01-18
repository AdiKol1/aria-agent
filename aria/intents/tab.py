"""Tab Intent Handler - Handles browser/app tab management.

This module provides the handler for tab-related actions including
creating, closing, and switching between tabs.
"""
import re
from typing import Dict, Any, List, Optional
from .base import IntentHandler, Intent, IntentResult, IntentType


class TabIntentHandler(IntentHandler):
    """Handler for tab intents.

    Recognizes and processes commands like:
    - "new tab"
    - "close this tab"
    - "switch to next tab"
    - "go to tab 3"
    - "previous tab"
    - "reopen last tab"
    """

    PATTERNS: List[str] = [
        r'^new\s+tab$',
        r'^open\s+(?:a\s+)?new\s+tab$',
        r'^create\s+(?:a\s+)?(?:new\s+)?tab$',
        r'^close\s+(?:this\s+)?tab$',
        r'^close\s+(?:the\s+)?current\s+tab$',
        r'^switch\s+(?:to\s+)?(?:the\s+)?next\s+tab$',
        r'^next\s+tab$',
        r'^switch\s+(?:to\s+)?(?:the\s+)?previous\s+tab$',
        r'^previous\s+tab$',
        r'^prev\s+tab$',
        r'^switch\s+(?:to\s+)?tab\s+(\d+)$',
        r'^go\s+to\s+tab\s+(\d+)$',
        r'^tab\s+(\d+)$',
        r'^(?:re)?open\s+(?:the\s+)?last\s+(?:closed\s+)?tab$',
        r'^restore\s+(?:the\s+)?last\s+tab$',
        r'^reopen\s+(?:closed\s+)?tab$',
        r'^first\s+tab$',
        r'^last\s+tab$',
        r'^close\s+all\s+(?:other\s+)?tabs?$',
        r'^close\s+tabs?\s+to\s+the\s+(left|right)$',
    ]

    # Keywords that strongly indicate a tab intent
    KEYWORDS: List[str] = [
        "tab", "new tab", "close tab", "next tab",
        "previous tab", "switch tab"
    ]

    # Tab action types
    TAB_ACTIONS = {
        "new": ["new tab", "open new tab", "create tab"],
        "close": ["close tab", "close this tab", "close current tab"],
        "next": ["next tab", "switch to next tab"],
        "previous": ["previous tab", "prev tab", "switch to previous tab"],
        "switch": ["switch to tab", "go to tab", "tab"],
        "reopen": ["reopen tab", "restore tab", "open last tab"],
        "first": ["first tab"],
        "last": ["last tab"],
    }

    # Keyboard shortcuts for tab actions (for reference)
    TAB_SHORTCUTS: Dict[str, List[str]] = {
        "new": ["command", "t"],
        "close": ["command", "w"],
        "next": ["command", "shift", "]"],
        "previous": ["command", "shift", "["],
        "reopen": ["command", "shift", "t"],
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
        if "tab" in text_lower:
            # High confidence if combined with action words
            action_words = ["new", "close", "open", "switch", "next", "previous", "prev", "go to", "reopen"]
            if any(word in text_lower for word in action_words):
                return 0.95
            return 0.7

        return 0.0

    def extract_params(self, text: str) -> Dict[str, Any]:
        """Extract parameters from the input text.

        Args:
            text: The user input text to parse.

        Returns:
            Dictionary containing:
            - action: The tab action (new, close, next, previous, switch, reopen)
            - tab_number: Tab number if switching to specific tab
            - direction: left/right for closing tabs to one side
        """
        if not text:
            return {}

        text_lower = text.lower().strip()
        params: Dict[str, Any] = {}

        # Determine the action
        action = self._determine_action(text_lower)
        if action:
            params["action"] = action

        # Extract tab number if switching to specific tab
        tab_match = re.search(r'tab\s+(\d+)', text_lower)
        if tab_match:
            params["tab_number"] = int(tab_match.group(1))
            params["action"] = "switch"

        # Handle "first tab" and "last tab"
        if "first tab" in text_lower:
            params["action"] = "first"
            params["tab_number"] = 1
        elif "last tab" in text_lower:
            params["action"] = "last"

        # Extract direction for closing tabs
        direction_match = re.search(r'to\s+the\s+(left|right)', text_lower)
        if direction_match:
            params["direction"] = direction_match.group(1)

        # Check for "all tabs" or "other tabs"
        if "all" in text_lower and "close" in text_lower:
            params["close_all"] = True
        if "other" in text_lower and "close" in text_lower:
            params["close_others"] = True

        return params

    def _determine_action(self, text: str) -> Optional[str]:
        """Determine the tab action from text.

        Args:
            text: The command text.

        Returns:
            The action type.
        """
        # Check for new tab
        if "new" in text or "create" in text or ("open" in text and "last" not in text):
            return "new"

        # Check for close tab
        if "close" in text:
            return "close"

        # Check for next tab
        if "next" in text:
            return "next"

        # Check for previous tab
        if "previous" in text or "prev" in text:
            return "previous"

        # Check for reopen/restore
        if "reopen" in text or "restore" in text or "last" in text:
            return "reopen"

        # Check for switch (with number)
        if "switch" in text or "go to" in text:
            return "switch"

        return None

    def _get_shortcut_for_action(self, action: str) -> Optional[List[str]]:
        """Get the keyboard shortcut for a tab action.

        Args:
            action: The tab action.

        Returns:
            List of keys for the shortcut.
        """
        return self.TAB_SHORTCUTS.get(action)

    def handle(self, intent: Intent) -> IntentResult:
        """Execute the tab intent.

        Note: This is a placeholder. The actual tab execution would be
        handled by the action executor which interfaces with keyboard shortcuts.

        Args:
            intent: The parsed tab intent.

        Returns:
            The result of the tab action.
        """
        action = intent.params.get("action", "unknown")
        tab_number = intent.params.get("tab_number")

        # Build response based on action
        action_messages = {
            "new": "Would open a new tab",
            "close": "Would close the current tab",
            "next": "Would switch to the next tab",
            "previous": "Would switch to the previous tab",
            "reopen": "Would reopen the last closed tab",
            "first": "Would switch to the first tab",
            "last": "Would switch to the last tab",
        }

        if action == "switch" and tab_number:
            response = f"Would switch to tab {tab_number}"
        elif action in action_messages:
            response = action_messages[action]
        else:
            response = "Would perform tab action"

        # Add info about close all/others
        if intent.params.get("close_all"):
            response = "Would close all tabs"
        elif intent.params.get("close_others"):
            response = "Would close all other tabs"
        elif intent.params.get("direction"):
            response = f"Would close tabs to the {intent.params['direction']}"

        # Get the shortcut if available
        shortcut = self._get_shortcut_for_action(action)

        return IntentResult.success_result(
            response=response,
            data={
                "action": f"tab_{action}",
                "tab_number": tab_number,
                "shortcut": shortcut,
                "params": intent.params
            }
        )
