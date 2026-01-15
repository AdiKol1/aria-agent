"""
Screen Capture and Claude Vision for Aria

Uses macOS ScreenCaptureKit for efficient screen capture
and Claude's vision capabilities for understanding.
"""

import base64
import io
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
from PIL import Image

from .config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_MAX_TOKENS,
    SCREENSHOTS_PATH,
    PRIVATE_APPS,
)


class ScreenCapture:
    """Handles screen capture on macOS."""

    def __init__(self):
        self.last_capture: Optional[Image.Image] = None
        self.last_capture_time: Optional[datetime] = None

    def get_active_app(self) -> str:
        """Get the name of the currently focused application."""
        script = '''
        tell application "System Events"
            set frontApp to name of first application process whose frontmost is true
        end tell
        return frontApp
        '''
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip()
        except Exception:
            return "Unknown"

    def is_private_app_focused(self) -> bool:
        """Check if a private app is currently focused."""
        active_app = self.get_active_app()
        return active_app in PRIVATE_APPS

    def capture(self, save: bool = False) -> Optional[Image.Image]:
        """
        Capture the current screen.

        Returns None if a private app is focused.
        """
        active_app = self.get_active_app()
        print(f"Active app: {active_app}")

        if self.is_private_app_focused():
            print("Private app focused, skipping capture")
            return None

        try:
            # Use screencapture command (reliable, no permissions issues if granted)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_path = SCREENSHOTS_PATH / f"capture_{timestamp}.png"

            print(f"Running screencapture to {temp_path}...")

            # -x: no sound, -C: capture cursor
            result = subprocess.run(
                ["screencapture", "-x", str(temp_path)],
                capture_output=True,
                timeout=10
            )

            if result.returncode != 0:
                print(f"screencapture failed: {result.stderr.decode()}")
                return None

            if temp_path.exists():
                print(f"Screenshot saved, size: {temp_path.stat().st_size} bytes")
                image = Image.open(temp_path)
                self.last_capture = image.copy()
                self.last_capture_time = datetime.now()

                if not save:
                    temp_path.unlink()  # Delete if not saving

                return image
            else:
                print("Screenshot file not created - check Screen Recording permission")
                return None

        except Exception as e:
            print(f"Screen capture error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def capture_to_base64(self, max_width: int = 1920) -> Optional[str]:
        """Capture screen and return as base64 string for Claude."""
        result = self.capture_to_base64_with_size(max_width)
        return result[0] if result else None

    def capture_to_base64_with_size(self, max_width: int = 1920) -> Optional[tuple]:
        """Capture screen and return (base64_string, (width, height))."""
        print("Capturing screen...")
        image = self.capture(save=False)
        if image is None:
            print("Screen capture returned None (may need Screen Recording permission)")
            return None

        print(f"Captured image: {image.width}x{image.height}")

        # Resize if too large (for API efficiency)
        if image.width > max_width:
            ratio = max_width / image.width
            new_size = (max_width, int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"Resized to: {new_size}")

        final_size = (image.width, image.height)

        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        b64 = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
        print(f"Base64 length: {len(b64)} chars, final size: {final_size}")
        return (b64, final_size)


class ClaudeVision:
    """Uses Claude to understand what's on screen."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.screen = ScreenCapture()

    def analyze_screen(self, question: str = "What is on the screen?") -> str:
        """
        Capture the screen and ask Claude about it.

        Args:
            question: What to ask Claude about the screen

        Returns:
            Claude's analysis of the screen
        """
        screenshot_b64 = self.screen.capture_to_base64()

        if screenshot_b64 is None:
            return "I can't see the screen right now (private app focused or capture failed)."

        try:
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=CLAUDE_MAX_TOKENS,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": screenshot_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": question,
                            },
                        ],
                    }
                ],
            )

            return response.content[0].text

        except Exception as e:
            return f"Error analyzing screen: {e}"

    def get_screen_context(self) -> str:
        """Get a brief description of what's currently on screen."""
        return self.analyze_screen(
            "Briefly describe what application is open and what the user appears to be doing. "
            "Be concise - 1-2 sentences max."
        )

    def find_element(self, description: str) -> Optional[dict]:
        """
        Find a UI element on screen by description.

        Args:
            description: What to find (e.g., "the submit button", "the search field")

        Returns:
            Dict with x, y coordinates if found, None otherwise
        """
        screenshot_b64 = self.screen.capture_to_base64()

        if screenshot_b64 is None:
            return None

        try:
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": screenshot_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": f"""Find the UI element: "{description}"

If you can find it, respond with ONLY a JSON object like this:
{{"found": true, "x": 500, "y": 300, "element": "Submit Button"}}

The x,y coordinates should be the CENTER of the element, in screen pixels.

If you cannot find it, respond with:
{{"found": false, "reason": "why not found"}}

Respond with ONLY the JSON, no other text.""",
                            },
                        ],
                    }
                ],
            )

            import json
            result = json.loads(response.content[0].text)
            return result if result.get("found") else None

        except Exception as e:
            print(f"Error finding element: {e}")
            return None


# Singleton instances
_screen_capture: Optional[ScreenCapture] = None
_claude_vision: Optional[ClaudeVision] = None


def get_screen_capture() -> ScreenCapture:
    """Get the singleton ScreenCapture instance."""
    global _screen_capture
    if _screen_capture is None:
        _screen_capture = ScreenCapture()
    return _screen_capture


def get_claude_vision() -> ClaudeVision:
    """Get the singleton ClaudeVision instance."""
    global _claude_vision
    if _claude_vision is None:
        _claude_vision = ClaudeVision()
    return _claude_vision
