"""
Desktop control specialist agent.
"""

from .base import BaseAgent, AgentContext, AgentResult

class SystemAgent(BaseAgent):
    """Handles desktop control: click, type, open apps, screenshots."""

    name = "system"
    description = "Handles desktop control like clicking, typing, opening apps, and taking screenshots"
    triggers = [
        "click", "type", "press", "scroll",
        "open app", "launch", "start", "run",
        "screenshot", "capture", "screen",
        "window", "minimize", "maximize", "close"
    ]

    def __init__(self):
        super().__init__()
        # Import control module
        try:
            from aria.control import get_control
            self.control = get_control()
        except ImportError:
            self.control = None

    async def process(self, context: AgentContext) -> AgentResult:
        """Process system/desktop control request."""
        if not self.control:
            return AgentResult.error("Desktop control not available")

        input_lower = context.user_input.lower()

        if "click" in input_lower:
            return await self._click(context)
        elif "type" in input_lower or "write" in input_lower:
            return await self._type_text(context)
        elif any(w in input_lower for w in ["open", "launch", "start", "run"]):
            return await self._open_app(context)
        elif any(w in input_lower for w in ["screenshot", "capture", "screen"]):
            return await self._screenshot(context)
        elif "scroll" in input_lower:
            return await self._scroll(context)
        elif "press" in input_lower or "key" in input_lower:
            return await self._press_key(context)
        else:
            # Hand off to default behavior
            return AgentResult.handoff("system", "I'll handle this with general desktop control.")

    async def _click(self, context: AgentContext) -> AgentResult:
        """Click at coordinates or on element."""
        import re
        match = re.search(r'(\d+)\s*,?\s*(\d+)', context.user_input)

        if match:
            x, y = int(match.group(1)), int(match.group(2))
            success = self.control.click(x, y)
            if success:
                return AgentResult.ok(f"Clicked at ({x}, {y})")
            else:
                return AgentResult.error(f"Failed to click at ({x}, {y})")
        else:
            return AgentResult.error("Please specify coordinates (e.g., 'click at 500, 300')")

    async def _type_text(self, context: AgentContext) -> AgentResult:
        """Type text."""
        # Extract text to type
        text = context.user_input
        for prefix in ["type", "write", "enter"]:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
                break

        text = text.strip('"\'')

        if not text:
            return AgentResult.error("What should I type?")

        success = self.control.type_text(text)
        if success:
            return AgentResult.ok(f"Typed: {text[:50]}...")
        else:
            return AgentResult.error("Failed to type text")

    async def _open_app(self, context: AgentContext) -> AgentResult:
        """Open an application."""
        import re
        match = re.search(r'(?:open|launch|start|run)\s+(\w+)', context.user_input, re.I)

        if match:
            app_name = match.group(1)
            # Map common names
            app_map = {
                "chrome": "Google Chrome",
                "safari": "Safari",
                "finder": "Finder",
                "terminal": "Terminal",
                "code": "Visual Studio Code",
                "vscode": "Visual Studio Code",
            }
            actual_name = app_map.get(app_name.lower(), app_name.title())

            success = self.control.open_app(actual_name)
            if success:
                return AgentResult.ok(f"Opened {actual_name}")
            else:
                return AgentResult.error(f"Failed to open {actual_name}")
        else:
            return AgentResult.error("Which app should I open?")

    async def _screenshot(self, context: AgentContext) -> AgentResult:
        """Take a screenshot."""
        try:
            from aria.vision import get_screen_capture
            screen = get_screen_capture()
            screenshot = screen.capture_to_base64_with_size()
            if screenshot:
                return AgentResult.ok("Screenshot captured", data={"screenshot": screenshot[0]})
            else:
                return AgentResult.error("Failed to capture screenshot")
        except Exception as e:
            return AgentResult.error(f"Screenshot error: {e}")

    async def _scroll(self, context: AgentContext) -> AgentResult:
        """Scroll up or down."""
        input_lower = context.user_input.lower()

        if "up" in input_lower:
            amount = 3
        elif "down" in input_lower:
            amount = -3
        else:
            amount = -3  # Default down

        success = self.control.scroll(amount)
        direction = "up" if amount > 0 else "down"
        if success:
            return AgentResult.ok(f"Scrolled {direction}")
        else:
            return AgentResult.error("Failed to scroll")

    async def _press_key(self, context: AgentContext) -> AgentResult:
        """Press a keyboard key."""
        input_lower = context.user_input.lower()

        key_map = {
            "enter": "enter", "return": "enter",
            "tab": "tab", "escape": "escape", "esc": "escape",
            "space": "space", "backspace": "backspace",
            "delete": "delete", "up": "up", "down": "down",
            "left": "left", "right": "right"
        }

        for key_name, key_value in key_map.items():
            if key_name in input_lower:
                success = self.control.press_key(key_value)
                if success:
                    return AgentResult.ok(f"Pressed {key_value}")
                else:
                    return AgentResult.error(f"Failed to press {key_value}")

        return AgentResult.error("Which key should I press?")
