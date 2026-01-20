"""
Claude Computer Use Agent Loop

Implements the Claude Computer Use pattern for autonomous computer control.
Claude drives everything: screenshot → analyze → act → verify → loop
"""

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

from anthropic import Anthropic

from .config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_MODEL_SIMPLE,
    CLAUDE_MODEL_COMPLEX,
    SCREENSHOT_MAX_WIDTH,
    USE_SMART_MODEL_SELECTION,
)
from .task_classifier import classify_task, get_model_for_task
from .control import get_control
from .vision import get_screen_capture


class ToolResult:
    """Result from executing a tool."""

    def __init__(
        self,
        output: str = "",
        error: Optional[str] = None,
        base64_image: Optional[str] = None,
    ):
        self.output = output
        self.error = error
        self.base64_image = base64_image


class ComputerTool:
    """
    Computer control tool for Claude.

    Handles mouse, keyboard, and screenshot operations.
    Mirrors the Anthropic computer use reference implementation.
    """

    name = "computer"

    def __init__(self):
        self.control = get_control()
        self.screen = get_screen_capture()
        self.screen_width, self.screen_height = self.control.screen_width, self.control.screen_height

    @property
    def tool_definition(self) -> dict:
        """Return the tool definition for the Claude API."""
        return {
            "type": "computer_20250124",
            "name": self.name,
            "display_width_px": min(self.screen_width, SCREENSHOT_MAX_WIDTH),
            "display_height_px": int(self.screen_height * (min(self.screen_width, SCREENSHOT_MAX_WIDTH) / self.screen_width)),
            "display_number": 1,
        }

    def __call__(self, action: str, **kwargs) -> ToolResult:
        """Execute a computer action."""
        try:
            if action == "screenshot":
                return self._screenshot()
            elif action == "mouse_move":
                return self._mouse_move(kwargs.get("coordinate", [0, 0]))
            elif action == "left_click":
                return self._click("left", kwargs.get("coordinate"))
            elif action == "right_click":
                return self._click("right", kwargs.get("coordinate"))
            elif action == "middle_click":
                return self._click("middle", kwargs.get("coordinate"))
            elif action == "double_click":
                return self._double_click(kwargs.get("coordinate"))
            elif action == "left_click_drag":
                return self._drag(kwargs.get("start_coordinate", [0, 0]), kwargs.get("end_coordinate", [0, 0]))
            elif action == "type":
                return self._type(kwargs.get("text", ""))
            elif action == "key":
                return self._key(kwargs.get("key", ""))
            elif action == "scroll":
                return self._scroll(kwargs.get("coordinate"), kwargs.get("delta", [0, 0]))
            elif action == "wait":
                return self._wait(kwargs.get("duration", 1))
            else:
                return ToolResult(error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(error=str(e))

    def _screenshot(self) -> ToolResult:
        """Take a screenshot and return as base64."""
        b64 = self.screen.capture_to_base64(max_width=SCREENSHOT_MAX_WIDTH)
        if b64:
            return ToolResult(output="Screenshot captured", base64_image=b64)
        return ToolResult(error="Failed to capture screenshot")

    def _scale_coordinates(self, coordinate: list) -> tuple:
        """Scale coordinates from API space to screen space."""
        if not coordinate or len(coordinate) < 2:
            return (0, 0)

        x, y = coordinate[0], coordinate[1]

        # Scale from API space (max SCREENSHOT_MAX_WIDTH) to actual screen
        if self.screen_width > SCREENSHOT_MAX_WIDTH:
            scale = self.screen_width / SCREENSHOT_MAX_WIDTH
            x = int(x * scale)
            y = int(y * scale)

        return (x, y)

    def _mouse_move(self, coordinate: list) -> ToolResult:
        """Move mouse to coordinates."""
        x, y = self._scale_coordinates(coordinate)
        success = self.control.move_to(x, y)
        if success:
            return ToolResult(output=f"Moved mouse to ({x}, {y})")
        return ToolResult(error=f"Failed to move mouse to ({x}, {y})")

    def _click(self, button: str, coordinate: Optional[list] = None) -> ToolResult:
        """Click at coordinates (or current position if None)."""
        if coordinate:
            x, y = self._scale_coordinates(coordinate)
            success = self.control.click(x, y, button=button)
        else:
            import pyautogui
            x, y = pyautogui.position()
            success = self.control.click(x, y, button=button)

        if success:
            return ToolResult(output=f"{button.capitalize()} clicked at ({x}, {y})")
        return ToolResult(error=f"Failed to {button} click at ({x}, {y})")

    def _double_click(self, coordinate: Optional[list] = None) -> ToolResult:
        """Double click at coordinates."""
        if coordinate:
            x, y = self._scale_coordinates(coordinate)
            success = self.control.double_click(x, y)
        else:
            import pyautogui
            x, y = pyautogui.position()
            success = self.control.double_click(x, y)

        if success:
            return ToolResult(output=f"Double clicked at ({x}, {y})")
        return ToolResult(error=f"Failed to double click at ({x}, {y})")

    def _drag(self, start: list, end: list) -> ToolResult:
        """Drag from start to end coordinates."""
        sx, sy = self._scale_coordinates(start)
        ex, ey = self._scale_coordinates(end)
        success = self.control.drag_to(sx, sy, ex, ey)
        if success:
            return ToolResult(output=f"Dragged from ({sx}, {sy}) to ({ex}, {ey})")
        return ToolResult(error=f"Failed to drag from ({sx}, {sy}) to ({ex}, {ey})")

    def _type(self, text: str) -> ToolResult:
        """Type text."""
        success = self.control.type_text(text)
        if success:
            return ToolResult(output=f"Typed: {text[:50]}{'...' if len(text) > 50 else ''}")
        return ToolResult(error=f"Failed to type text")

    def _key(self, key: str) -> ToolResult:
        """Press a key or key combination."""
        # Handle key combinations like "ctrl+c" or "cmd+space"
        if "+" in key:
            keys = key.lower().split("+")
            # Map common key names
            key_map = {
                "ctrl": "control",
                "cmd": "command",
                "meta": "command",
                "alt": "option",
                "win": "command",
                "return": "enter",
                "esc": "escape",
            }
            keys = [key_map.get(k.strip(), k.strip()) for k in keys]
            success = self.control.hotkey(*keys)
        else:
            # Map single keys
            key_map = {
                "return": "enter",
                "esc": "escape",
                "cmd": "command",
            }
            mapped_key = key_map.get(key.lower(), key)
            success = self.control.press_key(mapped_key)

        if success:
            return ToolResult(output=f"Pressed key: {key}")
        return ToolResult(error=f"Failed to press key: {key}")

    def _scroll(self, coordinate: Optional[list], delta: list) -> ToolResult:
        """Scroll at coordinates."""
        x, y = None, None
        if coordinate:
            x, y = self._scale_coordinates(coordinate)

        # delta[1] is vertical scroll (positive = up, negative = down)
        amount = delta[1] if len(delta) > 1 else delta[0] if delta else 0
        # Invert for pyautogui (positive = scroll up)
        amount = int(amount / 10)  # Scale down delta to reasonable scroll amount

        success = self.control.scroll(amount, x, y)
        if success:
            return ToolResult(output=f"Scrolled {amount} at ({x}, {y})")
        return ToolResult(error=f"Failed to scroll")

    def _wait(self, duration: float) -> ToolResult:
        """Wait for specified duration."""
        time.sleep(duration)
        return ToolResult(output=f"Waited {duration} seconds")


class BashTool:
    """
    Bash command execution tool for Claude.
    """

    name = "bash"

    @property
    def tool_definition(self) -> dict:
        """Return the tool definition for the Claude API."""
        return {
            "type": "bash_20250124",
            "name": self.name,
        }

    def __call__(self, command: str, restart: bool = False) -> ToolResult:
        """Execute a bash command."""
        import subprocess

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(__import__("pathlib").Path.home()),
            )

            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"

            if result.returncode != 0:
                return ToolResult(output=output, error=f"Exit code: {result.returncode}")

            return ToolResult(output=output or "(no output)")

        except subprocess.TimeoutExpired:
            return ToolResult(error="Command timed out after 120 seconds")
        except Exception as e:
            return ToolResult(error=str(e))


class ToolCollection:
    """Collection of tools available to Claude."""

    def __init__(self):
        self.computer = ComputerTool()
        self.bash = BashTool()
        self._tools = {
            "computer": self.computer,
            "bash": self.bash,
        }

    def get_tool_definitions(self) -> list:
        """Get all tool definitions for the API."""
        return [
            self.computer.tool_definition,
            self.bash.tool_definition,
        ]

    def execute(self, name: str, tool_input: dict) -> ToolResult:
        """Execute a tool by name."""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(error=f"Unknown tool: {name}")

        if name == "computer":
            action = tool_input.get("action", "")
            # Pass only the other kwargs, not action again
            kwargs = {k: v for k, v in tool_input.items() if k != "action"}
            return tool(action, **kwargs)
        elif name == "bash":
            command = tool_input.get("command", "")
            restart = tool_input.get("restart", False)
            return tool(command, restart)

        return ToolResult(error=f"Tool execution not implemented: {name}")


SYSTEM_PROMPT = """You are an AI assistant with direct control over a macOS computer. You can see the screen, move the mouse, click, type, and run commands.

CRITICAL PERFORMANCE RULES:
1. ALWAYS prefer bash commands over GUI interactions for speed:
   - Opening apps: `open -a "App Name"`
   - Opening URLs: `open -a "Google Chrome" "https://url"`
   - Switching apps: `osascript -e 'tell application "AppName" to activate'`
   - Typing in apps: Use AppleScript when possible

2. Only use GUI (mouse/keyboard) when bash won't work:
   - Clicking specific UI elements
   - Interacting with web page content
   - When bash commands fail

3. Take a screenshot ONLY when needed:
   - Before first action to see the state
   - After GUI actions to verify success
   - DON'T screenshot after bash commands (they're reliable)

4. If Terminal/another app steals focus, use bash to switch back:
   `osascript -e 'tell application "TARGET_APP" to activate'`

5. Be efficient - minimize iterations:
   - Combine related actions
   - Don't re-verify successful bash commands
   - Stop when task is complete

You have access to:
- computer tool: mouse, keyboard, screenshots
- bash tool: run terminal commands (PREFERRED for speed)

When the user asks you to do something:
1. Think about the fastest approach (bash vs GUI)
2. Execute with minimal screenshots
3. Only verify GUI actions
4. Report success when done"""


class ClaudeComputerUseAgent:
    """
    Claude Computer Use Agent.

    Implements the agentic loop:
    1. Take screenshot
    2. Send to Claude with task
    3. Execute tool calls
    4. Loop until complete
    """

    def __init__(
        self,
        on_message: Optional[Callable[[str], None]] = None,
        on_action: Optional[Callable[[str], None]] = None,
        max_iterations: int = 30,
    ):
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.tools = ToolCollection()
        self.on_message = on_message or print
        self.on_action = on_action or (lambda x: None)
        self.max_iterations = max_iterations
        self.messages = []
        self._last_bash_success = False  # Track if last action was successful bash
        self._selected_model = CLAUDE_MODEL  # Default model, may be overridden by smart selection

    def _make_api_call(self) -> Any:
        """Make an API call to Claude using the beta endpoint for computer use.

        Prompt caching reduces TTFT by ~50% for repeated prompts.
        The system prompt and tool definitions are cached server-side.
        """
        # Build system with cache control for the static parts
        system_content = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}  # Cache the system prompt
            }
        ]

        return self.client.beta.messages.create(
            model=self._selected_model,
            max_tokens=4096,
            system=system_content,
            tools=self.tools.get_tool_definitions(),
            messages=self.messages,
            betas=["computer-use-2025-01-24", "prompt-caching-2024-07-31"],
        )

    def _process_tool_call(self, tool_use: Any) -> dict:
        """Process a single tool call and return the result."""
        name = tool_use.name
        tool_input = tool_use.input

        self.on_action(f"Executing: {name}({json.dumps(tool_input)[:100]}...)")

        result = self.tools.execute(name, tool_input)

        # Track if this was a successful bash command (no screenshot needed after)
        self._last_bash_success = (name == "bash" and not result.error)

        # Build the tool result
        tool_result = {
            "type": "tool_result",
            "tool_use_id": tool_use.id,
        }

        if result.error:
            tool_result["content"] = result.error
            tool_result["is_error"] = True
        elif result.base64_image:
            tool_result["content"] = [
                {"type": "text", "text": result.output},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": result.base64_image,
                    },
                },
            ]
        else:
            # For successful bash commands, add hint to skip screenshot
            output = result.output
            if self._last_bash_success:
                output += "\n[Bash command succeeded - no screenshot verification needed]"
            tool_result["content"] = output

        return tool_result

    def run(self, task: str) -> str:
        """
        Run the agent loop for a given task.

        Args:
            task: The task to complete

        Returns:
            Final response from Claude
        """
        # Smart model selection based on task complexity
        if USE_SMART_MODEL_SELECTION:
            self._selected_model = get_model_for_task(
                task, CLAUDE_MODEL_SIMPLE, CLAUDE_MODEL_COMPLEX
            )
            classification, confidence = classify_task(task)
            self.on_action(
                f"Task classified as {classification} (confidence: {confidence:.1f}) -> using {self._selected_model}"
            )
        else:
            self._selected_model = CLAUDE_MODEL

        # Initialize with user task
        self.messages = [
            {
                "role": "user",
                "content": f"Please complete this task: {task}\n\nStart by taking a screenshot to see the current state.",
            }
        ]

        iteration = 0
        final_response = ""

        while iteration < self.max_iterations:
            iteration += 1
            self.on_action(f"--- Iteration {iteration} ---")

            # Call Claude
            try:
                response = self._make_api_call()
            except Exception as e:
                self.on_message(f"API Error: {e}")
                return f"Error: {e}"

            # Process the response
            assistant_content = []
            tool_results = []

            for block in response.content:
                if block.type == "text":
                    self.on_message(block.text)
                    final_response = block.text
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
                    tool_result = self._process_tool_call(block)
                    tool_results.append(tool_result)

            # Add assistant message
            self.messages.append({"role": "assistant", "content": assistant_content})

            # If no tool calls, we're done
            if not tool_results:
                break

            # Add tool results
            self.messages.append({"role": "user", "content": tool_results})

            # Check stop reason
            if response.stop_reason == "end_turn":
                break

        return final_response

    async def run_async(self, task: str) -> str:
        """Async version of run()."""
        return await asyncio.to_thread(self.run, task)


def create_agent(
    on_message: Optional[Callable[[str], None]] = None,
    on_action: Optional[Callable[[str], None]] = None,
) -> ClaudeComputerUseAgent:
    """Create a new Claude Computer Use agent."""
    return ClaudeComputerUseAgent(on_message=on_message, on_action=on_action)


# Simple CLI for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m aria.claude_computer_use 'your task here'")
        sys.exit(1)

    task = " ".join(sys.argv[1:])
    print(f"Task: {task}")
    print("=" * 50)

    agent = create_agent()
    result = agent.run(task)

    print("=" * 50)
    print(f"Final result: {result}")
