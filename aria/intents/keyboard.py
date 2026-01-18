"""Keyboard Intent Handler - Handles keyboard shortcuts and key presses.

This module provides the handler for keyboard shortcuts like copy/paste,
as well as individual key presses like Enter or Escape.
"""
import re
from typing import Dict, Any, List, Optional, Tuple
from .base import IntentHandler, Intent, IntentResult, IntentType


class KeyboardIntentHandler(IntentHandler):
    """Handler for keyboard intents.

    Recognizes and processes commands like:
    - "copy"
    - "paste"
    - "undo"
    - "press command+s"
    - "hit enter"
    - "do ctrl+c"
    - "select all"
    """

    PATTERNS: List[str] = [
        # Named shortcuts
        r'^copy$',
        r'^paste$',
        r'^cut$',
        r'^undo$',
        r'^redo$',
        r'^save$',
        r'^delete$',
        r'^find$',
        r'^search$',
        r'^select\s+all$',
        r'^new\s+(?:document|file|window)$',
        r'^close\s+(?:document|file|window)$',
        # Explicit shortcuts
        r'^(?:do\s+)?(?:press\s+)?(?:hit\s+)?(cmd|command|ctrl|control|alt|option|shift)[\s+\-]+(.+)$',
        r'^(?:press|hit)\s+(enter|return|escape|esc|tab|space|backspace|delete)$',
        r'^press\s+(.+)$',
    ]

    # Keywords that strongly indicate a keyboard intent
    KEYWORDS: List[str] = [
        "copy", "paste", "cut", "undo", "redo", "save", "delete",
        "select all", "find", "search", "cmd", "command", "ctrl",
        "control", "alt", "option", "shift", "press", "hit"
    ]

    # Named shortcuts to key combinations
    SHORTCUTS: Dict[str, List[str]] = {
        "copy": ["command", "c"],
        "paste": ["command", "v"],
        "cut": ["command", "x"],
        "undo": ["command", "z"],
        "redo": ["command", "shift", "z"],
        "save": ["command", "s"],
        "save as": ["command", "shift", "s"],
        "select all": ["command", "a"],
        "find": ["command", "f"],
        "search": ["command", "f"],
        "new": ["command", "n"],
        "new window": ["command", "n"],
        "new document": ["command", "n"],
        "new file": ["command", "n"],
        "close": ["command", "w"],
        "close window": ["command", "w"],
        "close document": ["command", "w"],
        "close file": ["command", "w"],
        "quit": ["command", "q"],
        "print": ["command", "p"],
        "bold": ["command", "b"],
        "italic": ["command", "i"],
        "underline": ["command", "u"],
        "refresh": ["command", "r"],
        "reload": ["command", "r"],
        "zoom in": ["command", "="],
        "zoom out": ["command", "-"],
        "actual size": ["command", "0"],
        "minimize": ["command", "m"],
        "hide": ["command", "h"],
        "hide others": ["command", "option", "h"],
        "full screen": ["command", "control", "f"],
        "spotlight": ["command", "space"],
        "screenshot": ["command", "shift", "3"],
        "screenshot selection": ["command", "shift", "4"],
    }

    # Key name mappings
    KEY_ALIASES: Dict[str, str] = {
        "cmd": "command",
        "ctrl": "control",
        "ctl": "control",
        "opt": "option",
        "alt": "option",
        "return": "enter",
        "esc": "escape",
        "del": "delete",
        "backspace": "delete",
        "space": "space",
        "spacebar": "space",
    }

    # Single key presses
    SINGLE_KEYS: List[str] = [
        "enter", "return", "escape", "esc", "tab", "space",
        "backspace", "delete", "up", "down", "left", "right",
        "home", "end", "page up", "page down",
        "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12"
    ]

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

        # Check for named shortcuts (high confidence)
        if text_lower in self.SHORTCUTS:
            return 1.0

        # Check for pattern matches
        for pattern in self.PATTERNS:
            match = re.match(pattern, text_lower, re.IGNORECASE)
            if match:
                return 0.95

        # Check for keyboard-related keywords
        for keyword in self.KEYWORDS:
            if keyword in text_lower:
                return 0.8

        return 0.0

    def extract_params(self, text: str) -> Dict[str, Any]:
        """Extract parameters from the input text.

        Args:
            text: The user input text to parse.

        Returns:
            Dictionary containing:
            - keys: List of keys to press
            - shortcut_name: Name of the shortcut if recognized
        """
        if not text:
            return {}

        text_lower = text.lower().strip()
        params: Dict[str, Any] = {}

        # Check for named shortcuts first
        if text_lower in self.SHORTCUTS:
            params["keys"] = self.SHORTCUTS[text_lower]
            params["shortcut_name"] = text_lower
            return params

        # Check for "select all" as two words
        if "select all" in text_lower:
            params["keys"] = self.SHORTCUTS["select all"]
            params["shortcut_name"] = "select all"
            return params

        # Try to parse explicit shortcuts (cmd+key, ctrl+key, etc.)
        keys = self._parse_shortcut(text_lower)
        if keys:
            params["keys"] = keys
            return params

        # Check for single key presses
        single_key = self._extract_single_key(text_lower)
        if single_key:
            params["keys"] = [single_key]
            params["single_key"] = True
            return params

        return params

    def _parse_shortcut(self, text: str) -> Optional[List[str]]:
        """Parse an explicit keyboard shortcut.

        Args:
            text: The command text.

        Returns:
            List of keys in the shortcut.
        """
        # Match patterns like "cmd+c", "command-s", "ctrl z"
        shortcut_pattern = r'(?:do\s+)?(?:press\s+)?(?:hit\s+)?(cmd|command|ctrl|control|alt|option|shift)[\s+\-]+(.+)'
        match = re.search(shortcut_pattern, text)

        if not match:
            return None

        modifier = match.group(1).lower()
        key_part = match.group(2).lower().strip()

        # Normalize modifier
        modifier = self.KEY_ALIASES.get(modifier, modifier)

        keys = [modifier]

        # Parse the key part (might have more modifiers or just a key)
        # Handle cases like "shift+z" or just "z"
        parts = re.split(r'[\s+\-]+', key_part)
        for part in parts:
            part = part.strip()
            if part:
                normalized = self.KEY_ALIASES.get(part, part)
                if normalized not in keys:
                    keys.append(normalized)

        return keys

    def _extract_single_key(self, text: str) -> Optional[str]:
        """Extract a single key press from text.

        Args:
            text: The command text.

        Returns:
            The key to press.
        """
        # Remove common action words
        text = re.sub(r'^(?:press|hit|tap)\s+', '', text)

        for key in self.SINGLE_KEYS:
            if key in text:
                return self.KEY_ALIASES.get(key, key)

        return None

    def handle(self, intent: Intent) -> IntentResult:
        """Execute the keyboard intent.

        Note: This is a placeholder. The actual keyboard execution would be
        handled by the action executor which interfaces with the MCP tools.

        Args:
            intent: The parsed keyboard intent.

        Returns:
            The result of the keyboard action.
        """
        keys = intent.params.get("keys", [])

        if not keys:
            return IntentResult.error_result(
                error="No keys specified",
                response="I need to know which keys to press."
            )

        shortcut_name = intent.params.get("shortcut_name")

        if shortcut_name:
            return IntentResult.success_result(
                response=f"Would perform '{shortcut_name}' ({'+'.join(keys)})",
                data={
                    "action": "keyboard_shortcut",
                    "keys": keys,
                    "shortcut_name": shortcut_name,
                    "params": intent.params
                }
            )

        if intent.params.get("single_key"):
            return IntentResult.success_result(
                response=f"Would press '{keys[0]}'",
                data={
                    "action": "key_press",
                    "keys": keys,
                    "params": intent.params
                }
            )

        return IntentResult.success_result(
            response=f"Would press '{'+'.join(keys)}'",
            data={
                "action": "hotkey",
                "keys": keys,
                "params": intent.params
            }
        )
