"""
Aria Agent Brain

The core reasoning engine that combines vision, voice, and control.
"""

import json
from typing import Optional, List, Dict, Any

import anthropic

from .config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MAX_TOKENS
from .vision import get_claude_vision, get_screen_capture
from .control import get_control
from .memory import get_memory
from .claude_bridge import get_claude_bridge


ARIA_SYSTEM_PROMPT = """You are Aria, a voice assistant that controls a Mac computer. You can see the screen and MUST take action.

## MEMORY
You have long-term memory! Information in [MEMORY CONTEXT] blocks is what you remember about the user.
- Use this context to personalize your responses
- Reference past interactions when relevant ("Last time we did X...")
- Apply learned preferences ("I know you prefer...")
- The user can teach you things and you'll remember them

## CRITICAL RULES
1. Keep responses to 1 SHORT sentence
2. When asked to do something, ALWAYS include the action JSON
3. Never just say you'll do something - actually do it with JSON
4. Use your memory context when it's relevant to the request

## Actions - YOU MUST USE THESE
When the user asks you to click, type, or interact, include a JSON code block:

```json
{"action": "click", "x": 500, "y": 300}
```

Available actions:
- click: {"action": "click", "x": X, "y": Y}
- type: {"action": "type", "text": "text here"}
- press: {"action": "press", "key": "enter"}
- hotkey: {"action": "hotkey", "keys": ["command", "w"]}
- scroll: {"action": "scroll", "amount": -3}
- open_app: {"action": "open_app", "app": "Safari"}
- open_url: {"action": "open_url", "url": "https://facebook.com"}

## NAVIGATING TO URLs - ALWAYS USE open_url:
ALWAYS use open_url action for ANY website navigation - it's instant and reliable:
```json
{"action": "open_url", "url": "https://facebook.com"}
```

Examples:
- "Go to Facebook" → {"action": "open_url", "url": "https://facebook.com"}
- "Open Google" → {"action": "open_url", "url": "https://google.com"}
- "Navigate to twitter" → {"action": "open_url", "url": "https://twitter.com"}

DO NOT try to click the address bar and type - just use open_url!

## CORRECT Examples:

User: "Close this dialog"
You: "Closing it now.
```json
{"action": "click", "x": 845, "y": 520}
```"

User: "Open Safari"
You: "Opening Safari.
```json
{"action": "open_app", "app": "Safari"}
```"

User: "Press enter"
You: "Done.
```json
{"action": "press", "key": "enter"}
```"

## WRONG - Never do this:
- "I'll close that for you" (no action JSON = nothing happens!)
- "Let me click that button" (no action JSON = nothing happens!)

## When you see a screenshot
- Look at coordinates carefully
- Estimate the x,y position of buttons/elements
- Top-left is (0,0), coordinates increase right and down
- A typical screen is around 1920x1080 or similar

REMEMBER: If you don't include the JSON action block, NOTHING will happen!
"""


class AriaAgent:
    """The Aria agent brain."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.vision = get_claude_vision()
        self.screen = get_screen_capture()
        self.control = get_control()
        self.memory = get_memory()
        self.claude_bridge = get_claude_bridge()
        self.conversation_history: List[Dict[str, Any]] = []

        # For coordinate scaling (Retina displays)
        # pyautogui uses logical coordinates, screenshots are physical pixels
        import pyautogui
        self.logical_screen_size = pyautogui.size()  # e.g., (1710, 1107)
        self.image_size_for_claude = (1920, 1242)  # What we send to Claude
        print(f"Screen size (logical): {self.logical_screen_size}")
        print(f"Memory loaded: {len(self.memory.get_all_facts())} facts remembered")
        print(f"Claude Code bridge ready (project: {self.claude_bridge.current_project})")

    def process_request(self, user_input: str, include_screen: bool = True) -> str:
        """
        Process a user request.

        Args:
            user_input: What the user said/asked
            include_screen: Whether to include a screenshot for context

        Returns:
            Text response to speak to user
        """
        # Check if this is a coding request - delegate to Claude Code
        if self.claude_bridge.is_coding_request(user_input):
            print(f"Detected coding request, delegating to Claude Code...")
            return self._handle_coding_request(user_input)

        # Build the message content
        content = []

        # Include screenshot if requested
        if include_screen:
            print("Attempting screen capture...")
            screenshot_b64, image_size = self.screen.capture_to_base64_with_size()
            if screenshot_b64:
                self.image_size_for_claude = image_size  # Update the actual size
                print(f"Screenshot captured! Size: {image_size}, Adding to message ({len(screenshot_b64)} chars)")
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": screenshot_b64
                    }
                })
            else:
                print("WARNING: Screenshot capture failed - Claude won't see the screen")

        # Get relevant memories
        memory_context = self.memory.get_context_for_request(user_input)

        # Add user text with screen size info and memories
        screen_info = ""
        if include_screen and screenshot_b64:
            screen_info = f"\n[Screen size: {self.image_size_for_claude[0]}x{self.image_size_for_claude[1]} pixels]"

        # Build the full user message with memory context
        full_message = user_input + screen_info
        if memory_context:
            full_message = f"[MEMORY CONTEXT]\n{memory_context}\n[END MEMORY]\n\n{full_message}"
            print(f"Added memory context: {memory_context[:100]}...")

        content.append({
            "type": "text",
            "text": full_message
        })

        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": content
        })

        # Trim history if too long (keep last 10 exchanges)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        try:
            # Call Claude with lower max tokens for snappier responses
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=300,  # Keep responses short for voice
                system=ARIA_SYSTEM_PROMPT,
                messages=self.conversation_history
            )

            assistant_message = response.content[0].text
            print(f"Claude raw response: {assistant_message}")

            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            # Parse and execute any actions
            text_response, actions = self._parse_response(assistant_message)

            if actions:
                print(f"Found {len(actions)} actions to execute: {actions}")
                for i, action in enumerate(actions):
                    print(f"Executing action {i+1}/{len(actions)}: {action}")
                    success = self._execute_action(action)
                    print(f"Action result: {'success' if success else 'failed'}")
                    # Add delay between actions for UI to respond
                    if i < len(actions) - 1:
                        import time
                        time.sleep(0.5)
            else:
                print("No actions found in response - Claude may need clearer instructions")

            # Extract and store memories in background
            import threading
            threading.Thread(
                target=self.memory.extract_and_store_memories,
                args=(user_input, text_response, actions),
                daemon=True
            ).start()

            return text_response

        except Exception as e:
            error_msg = f"I encountered an error: {str(e)[:100]}"
            print(f"Agent error: {e}")
            return error_msg

    def _parse_response(self, response: str) -> tuple[str, List[dict]]:
        """Parse response text and extract any action JSON blocks."""
        actions = []
        text_parts = []

        # Split by code blocks
        parts = response.split("```")

        for i, part in enumerate(parts):
            if i % 2 == 0:
                # Text part
                text_parts.append(part.strip())
            else:
                # Code block - check if JSON
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    action = json.loads(part)
                    if isinstance(action, dict) and "action" in action:
                        actions.append(action)
                except json.JSONDecodeError:
                    # Not valid JSON, treat as text
                    text_parts.append(part)

        text_response = " ".join(filter(None, text_parts))
        return text_response, actions

    def _scale_coordinates(self, x: int, y: int) -> tuple:
        """Scale coordinates from Claude's image space to logical screen space."""
        # Claude sees a 1920xH image, but pyautogui uses logical screen coordinates
        scale_x = self.logical_screen_size[0] / self.image_size_for_claude[0]
        scale_y = self.logical_screen_size[1] / self.image_size_for_claude[1]

        scaled_x = int(x * scale_x)
        scaled_y = int(y * scale_y)

        print(f"Scaling coordinates: ({x}, {y}) -> ({scaled_x}, {scaled_y})")
        print(f"  Scale factors: x={scale_x:.2f}, y={scale_y:.2f}")

        return scaled_x, scaled_y

    def _execute_action(self, action: dict) -> bool:
        """Execute a single action."""
        action_type = action.get("action")

        try:
            if action_type == "click":
                x, y = self._scale_coordinates(action["x"], action["y"])
                return self.control.click(x, y)

            elif action_type == "double_click":
                x, y = self._scale_coordinates(action["x"], action["y"])
                return self.control.double_click(x, y)

            elif action_type == "right_click":
                x, y = self._scale_coordinates(action["x"], action["y"])
                return self.control.right_click(x, y)

            elif action_type == "type":
                return self.control.type_text(action["text"])

            elif action_type == "press":
                return self.control.press_key(action["key"])

            elif action_type == "hotkey":
                return self.control.hotkey(*action["keys"])

            elif action_type == "scroll":
                x = action.get("x")
                y = action.get("y")
                return self.control.scroll(action["amount"], x, y)

            elif action_type == "open_app":
                return self.control.open_app(action["app"])

            elif action_type == "open_url":
                return self.control.open_url(action["url"])

            elif action_type == "wait":
                import time
                time.sleep(action.get("seconds", 1))
                return True

            else:
                print(f"Unknown action type: {action_type}")
                return False

        except Exception as e:
            print(f"Action execution error: {e}")
            return False

    def _handle_coding_request(self, user_input: str) -> str:
        """
        Handle a coding request by delegating to Claude Code.

        Args:
            user_input: The coding-related request

        Returns:
            Voice-friendly summary of what Claude Code did
        """
        try:
            # Run Claude Code
            print(f"Running Claude Code with: {user_input}")
            output = self.claude_bridge.run_claude(user_input)
            print(f"Claude Code output: {output[:200]}...")

            # Summarize for voice
            summary = self.claude_bridge.summarize_for_voice(output)

            # Store in memory
            self.memory.remember_interaction(
                f"Coding: {user_input[:50]}",
                user_input,
                summary
            )

            return summary

        except Exception as e:
            print(f"Claude Code error: {e}")
            return f"Sorry, I couldn't complete that coding task. Error: {str(e)[:50]}"

    def set_project(self, path: str) -> bool:
        """Set the current project for Claude Code."""
        return self.claude_bridge.set_project(path)

    def get_screen_context(self) -> str:
        """Get current screen context for voice response."""
        return self.vision.get_screen_context()

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []


# Singleton
_agent: Optional[AriaAgent] = None


def get_agent() -> AriaAgent:
    """Get the singleton AriaAgent instance."""
    global _agent
    if _agent is None:
        _agent = AriaAgent()
    return _agent
