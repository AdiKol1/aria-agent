"""Open Intent Handler - Handles app and file opening actions.

This module provides the handler for opening applications, files,
and other resources.
"""
import re
from typing import Dict, Any, List, Optional
from .base import IntentHandler, Intent, IntentResult, IntentType


class OpenIntentHandler(IntentHandler):
    """Handler for open intents.

    Recognizes and processes commands like:
    - "open Chrome"
    - "launch Terminal"
    - "start Finder"
    - "run the app"
    - "fire up Safari"
    - "bring up the calculator"
    """

    PATTERNS: List[str] = [
        r'^open\s+(?:the\s+)?(?:app\s+)?(.+)$',
        r'^launch\s+(?:the\s+)?(?:app\s+)?(.+)$',
        r'^start\s+(?:the\s+)?(?:app\s+)?(.+)$',
        r'^run\s+(?:the\s+)?(?:app\s+)?(.+)$',
        r'^open\s+up\s+(.+)$',
        r'^fire\s+up\s+(.+)$',
        r'^bring\s+up\s+(.+)$',
        r'^execute\s+(.+)$',
    ]

    # Keywords that strongly indicate an open intent
    KEYWORDS: List[str] = ["open", "launch", "start", "run", "fire up", "bring up"]

    # Common application name mappings (aliases to canonical names)
    APP_ALIASES: Dict[str, str] = {
        "chrome": "Google Chrome",
        "google chrome": "Google Chrome",
        "safari": "Safari",
        "firefox": "Firefox",
        "ff": "Firefox",
        "terminal": "Terminal",
        "term": "Terminal",
        "iterm": "iTerm",
        "finder": "Finder",
        "files": "Finder",
        "vscode": "Visual Studio Code",
        "vs code": "Visual Studio Code",
        "code": "Visual Studio Code",
        "slack": "Slack",
        "spotify": "Spotify",
        "discord": "Discord",
        "zoom": "zoom.us",
        "teams": "Microsoft Teams",
        "word": "Microsoft Word",
        "excel": "Microsoft Excel",
        "powerpoint": "Microsoft PowerPoint",
        "outlook": "Microsoft Outlook",
        "notes": "Notes",
        "calendar": "Calendar",
        "mail": "Mail",
        "messages": "Messages",
        "imessage": "Messages",
        "facetime": "FaceTime",
        "photos": "Photos",
        "music": "Music",
        "itunes": "Music",
        "podcasts": "Podcasts",
        "tv": "TV",
        "apple tv": "TV",
        "books": "Books",
        "preview": "Preview",
        "calculator": "Calculator",
        "calc": "Calculator",
        "activity monitor": "Activity Monitor",
        "system preferences": "System Preferences",
        "settings": "System Preferences",
        "preferences": "System Preferences",
        "app store": "App Store",
        "maps": "Maps",
        "weather": "Weather",
        "news": "News",
        "stocks": "Stocks",
        "home": "Home",
        "reminders": "Reminders",
        "keynote": "Keynote",
        "pages": "Pages",
        "numbers": "Numbers",
        "xcode": "Xcode",
        "simulator": "Simulator",
        "android studio": "Android Studio",
        "intellij": "IntelliJ IDEA",
        "pycharm": "PyCharm",
        "webstorm": "WebStorm",
        "sublime": "Sublime Text",
        "atom": "Atom",
        "postman": "Postman",
        "docker": "Docker",
        "notion": "Notion",
        "figma": "Figma",
        "sketch": "Sketch",
        "photoshop": "Adobe Photoshop",
        "illustrator": "Adobe Illustrator",
        "premiere": "Adobe Premiere Pro",
        "after effects": "Adobe After Effects",
        "lightroom": "Adobe Lightroom",
        "obs": "OBS",
        "vlc": "VLC",
        "handbrake": "HandBrake",
        "1password": "1Password",
        "bitwarden": "Bitwarden",
        "lastpass": "LastPass",
        "alfred": "Alfred",
        "raycast": "Raycast",
        "bartender": "Bartender",
        "cleanmymac": "CleanMyMac",
        "dropbox": "Dropbox",
        "google drive": "Google Drive",
        "onedrive": "OneDrive",
        "transmit": "Transmit",
        "cyberduck": "Cyberduck",
        "filezilla": "FileZilla",
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
            if text_lower.startswith(keyword):
                return 0.9
            if f" {keyword}" in text_lower:
                return 0.5

        return 0.0

    def extract_params(self, text: str) -> Dict[str, Any]:
        """Extract parameters from the input text.

        Args:
            text: The user input text to parse.

        Returns:
            Dictionary containing:
            - app_name: The application to open
            - canonical_name: Normalized app name if available
            - file_path: Path if opening a file
        """
        if not text:
            return {}

        text_lower = text.lower().strip()
        params: Dict[str, Any] = {}

        # Extract target app/file
        target = self._extract_target(text_lower)
        if target:
            params["app_name"] = target

            # Try to get canonical name
            canonical = self._get_canonical_name(target)
            if canonical:
                params["canonical_name"] = canonical

        # Check if this looks like a file path
        if target and self._is_file_path(target):
            params["file_path"] = target
            params["is_file"] = True

        return params

    def _extract_target(self, text: str) -> Optional[str]:
        """Extract the open target from text.

        Args:
            text: The command text.

        Returns:
            The target application or file name.
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

        target = target.strip()

        # Remove common suffixes
        suffixes = [" app", " application", " please", " for me"]
        for suffix in suffixes:
            if target.lower().endswith(suffix):
                target = target[:-len(suffix)].strip()

        # Remove leading articles
        prefixes = ["the ", "a ", "an "]
        for prefix in prefixes:
            if target.lower().startswith(prefix):
                target = target[len(prefix):].strip()

        return target

    def _get_canonical_name(self, app_name: str) -> Optional[str]:
        """Get the canonical application name from an alias.

        Args:
            app_name: The app name or alias.

        Returns:
            The canonical application name, or None if not found.
        """
        app_lower = app_name.lower().strip()
        return self.APP_ALIASES.get(app_lower)

    def _is_file_path(self, target: str) -> bool:
        """Check if the target looks like a file path.

        Args:
            target: The target string.

        Returns:
            True if it appears to be a file path.
        """
        # Check for common file path indicators
        path_indicators = [
            target.startswith("/"),
            target.startswith("~"),
            target.startswith("./"),
            target.startswith("../"),
            "." in target and "/" in target,
            target.endswith(".app"),
            target.endswith(".txt"),
            target.endswith(".pdf"),
            target.endswith(".doc"),
            target.endswith(".docx"),
        ]
        return any(path_indicators)

    def handle(self, intent: Intent) -> IntentResult:
        """Execute the open intent.

        Note: This is a placeholder. The actual open execution would be
        handled by the action executor which interfaces with the MCP tools.

        Args:
            intent: The parsed open intent.

        Returns:
            The result of the open action.
        """
        target = intent.target or intent.params.get("app_name", "application")
        canonical = intent.params.get("canonical_name", target)

        if intent.params.get("is_file"):
            return IntentResult.success_result(
                response=f"Would open file '{target}'",
                data={
                    "action": "open_file",
                    "file_path": target,
                    "params": intent.params
                }
            )

        return IntentResult.success_result(
            response=f"Would open '{canonical}'",
            data={
                "action": "open_app",
                "app_name": canonical,
                "params": intent.params
            }
        )
