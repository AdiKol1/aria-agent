"""
Vision-Guided Action Executor for Aria.

This module provides screen-aware action execution with verification.
Instead of blindly clicking coordinates, it:
1. Captures the screen before acting
2. Uses Claude vision to understand what's visible
3. Determines precise coordinates for the target
4. Executes the action
5. Verifies the action succeeded
6. Retries or adapts if needed
"""

import json
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

from .config import ANTHROPIC_API_KEY, CLAUDE_MODEL, SCREENSHOT_MAX_WIDTH
from .lazy_anthropic import get_client as get_anthropic_client
from .vision import ScreenCapture
from .control import ComputerControl


@dataclass
class ActionResult:
    """Result of an action execution."""
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    retry_suggested: bool = False


class VisionActionExecutor:
    """Executes actions with visual verification using Claude."""

    def __init__(self, control: ComputerControl, screen: Optional[ScreenCapture] = None):
        """Initialize the action executor.

        Args:
            control: ComputerControl instance for executing actions.
            screen: ScreenCapture instance (created if not provided).
        """
        self.control = control
        self.screen = screen or ScreenCapture()
        self.client = get_anthropic_client(ANTHROPIC_API_KEY)
        self.max_retries = 3
        self.last_screen_state: Optional[str] = None

        # Screen caching to avoid redundant captures
        self._screen_cache: Optional[Tuple[str, Tuple[int, int]]] = None
        self._screen_cache_time: float = 0
        self._screen_cache_ttl: float = 2.0  # Cache valid for 2 seconds

    def _capture_screen_b64(self, use_cache: bool = True) -> Optional[Tuple[str, Tuple[int, int]]]:
        """Capture screen and return (base64, (width, height)).

        Uses caching to avoid redundant captures within TTL window.
        """
        now = time.time()

        # Return cached if still valid
        if use_cache and self._screen_cache and (now - self._screen_cache_time) < self._screen_cache_ttl:
            print("[ActionExecutor] Using cached screen capture")
            return self._screen_cache

        # Capture new screen
        result = self.screen.capture_to_base64_with_size()
        if result:
            self._screen_cache = result
            self._screen_cache_time = now
        return result

    def invalidate_screen_cache(self):
        """Invalidate screen cache after an action that changes the screen."""
        self._screen_cache = None

    def _ask_claude_vision(self, image_b64: str, prompt: str) -> str:
        """Ask Claude to analyze an image and respond."""
        try:
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )
            return response.content[0].text
        except Exception as e:
            return f"Error analyzing image: {e}"

    def find_element(self, description: str) -> Optional[Dict[str, Any]]:
        """Find a UI element on screen by description.

        Args:
            description: What to find (e.g., "the File menu", "the search box")

        Returns:
            Dict with x, y coordinates (SCALED to actual screen) and element info, or None if not found.
        """
        result = self._capture_screen_b64()
        if not result:
            return None

        image_b64, (screenshot_width, screenshot_height) = result

        prompt = f"""Find the UI element: "{description}"

Screen size: {screenshot_width}x{screenshot_height} pixels

If you can find it, respond with ONLY a JSON object like this:
{{"found": true, "x": 500, "y": 300, "element": "File menu button", "confidence": "high"}}

The x,y coordinates should be the CENTER of the element, in screen pixels.
Be precise - look carefully at the actual position of the element.

If you cannot find it, respond with:
{{"found": false, "reason": "why not found", "suggestions": ["alternative approach"]}}

Respond with ONLY the JSON, no other text."""

        response = self._ask_claude_vision(image_b64, prompt)
        print(f"[ActionExecutor] Claude Vision response for '{description}': {response[:200]}...")

        try:
            # Extract JSON from response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            result = json.loads(response)

            if result.get("found"):
                # Scale coordinates from screenshot space to actual screen space
                # Screenshots may be resized to SCREENSHOT_MAX_WIDTH
                screen_width = self.control.screen_width
                screen_height = self.control.screen_height

                # Calculate scale factor
                if screen_width > SCREENSHOT_MAX_WIDTH:
                    scale_x = screen_width / screenshot_width
                    scale_y = screen_height / screenshot_height
                else:
                    scale_x = 1.0
                    scale_y = 1.0

                # Scale the coordinates
                original_x, original_y = result.get("x", 0), result.get("y", 0)
                result["x"] = int(original_x * scale_x)
                result["y"] = int(original_y * scale_y)
                result["original_coords"] = [original_x, original_y]
                result["scale_factor"] = [scale_x, scale_y]

                print(f"[ActionExecutor] Found '{description}' at ({original_x}, {original_y}) -> scaled to ({result['x']}, {result['y']})")
                return result
            else:
                # Log why element wasn't found
                reason = result.get("reason", "unknown")
                suggestions = result.get("suggestions", [])
                print(f"[ActionExecutor] Could not find '{description}': {reason}")
                if suggestions:
                    print(f"[ActionExecutor] Suggestions: {suggestions}")
                return None
        except json.JSONDecodeError:
            print(f"Failed to parse element location: {response}")
            return None

    def verify_action(self, expected_result: str) -> bool:
        """Verify that an action achieved the expected result.

        Args:
            expected_result: Description of what should be visible now.

        Returns:
            True if the expected result is visible.
        """
        result = self._capture_screen_b64()
        if not result:
            return False

        image_b64, _ = result

        prompt = f"""Look at this screen and determine if the following is true:
"{expected_result}"

Respond with ONLY a JSON object:
{{"verified": true/false, "observation": "what you see", "confidence": "high/medium/low"}}

Be accurate - only say verified:true if you can clearly see the expected result."""

        response = self._ask_claude_vision(image_b64, prompt)

        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            result = json.loads(response)
            return result.get("verified", False)
        except json.JSONDecodeError:
            return False

    def click_element(self, description: str, verify_after: Optional[str] = None) -> ActionResult:
        """Click on a UI element by description with verification.

        Args:
            description: What to click (e.g., "the File menu")
            verify_after: Optional description of expected result after clicking.

        Returns:
            ActionResult with success status and details.
        """
        for attempt in range(self.max_retries):
            # Find the element
            element = self.find_element(description)

            if not element:
                if attempt < self.max_retries - 1:
                    time.sleep(0.5)
                    continue
                return ActionResult(
                    success=False,
                    message=f"Could not find '{description}' on screen",
                    retry_suggested=True
                )

            x, y = element["x"], element["y"]
            element_name = element.get("element", description)

            # Execute the click
            print(f"[ActionExecutor] Clicking '{element_name}' at ({x}, {y})")
            self.control.click(x, y)

            # Wait for UI to respond
            time.sleep(0.3)

            # Verify if requested
            if verify_after:
                if self.verify_action(verify_after):
                    return ActionResult(
                        success=True,
                        message=f"Clicked '{element_name}' and verified: {verify_after}",
                        details={"x": x, "y": y, "element": element_name}
                    )
                else:
                    if attempt < self.max_retries - 1:
                        print(f"[ActionExecutor] Verification failed, retrying...")
                        time.sleep(0.5)
                        continue
                    return ActionResult(
                        success=False,
                        message=f"Clicked '{element_name}' but verification failed: {verify_after}",
                        details={"x": x, "y": y},
                        retry_suggested=True
                    )
            else:
                return ActionResult(
                    success=True,
                    message=f"Clicked '{element_name}' at ({x}, {y})",
                    details={"x": x, "y": y, "element": element_name}
                )

        return ActionResult(
            success=False,
            message=f"Failed to click '{description}' after {self.max_retries} attempts",
            retry_suggested=False
        )

    def open_menu_item(self, menu_name: str, item_name: str) -> ActionResult:
        """Open a menu and click an item.

        Args:
            menu_name: The menu to open (e.g., "File")
            item_name: The menu item to click (e.g., "New Window")

        Returns:
            ActionResult with success status.
        """
        # First, click the menu
        menu_result = self.click_element(
            f"the '{menu_name}' menu in the menu bar",
            verify_after=f"a dropdown menu is open showing menu items"
        )

        if not menu_result.success:
            return ActionResult(
                success=False,
                message=f"Could not open '{menu_name}' menu: {menu_result.message}",
                retry_suggested=True
            )

        # Wait for menu to open
        time.sleep(0.2)

        # Now click the menu item
        item_result = self.click_element(
            f"the '{item_name}' option in the dropdown menu",
            verify_after=None  # Verification depends on the action
        )

        if not item_result.success:
            # Close the menu by pressing Escape
            self.control.press_key("escape")
            return ActionResult(
                success=False,
                message=f"Could not click '{item_name}' in menu: {item_result.message}",
                retry_suggested=True
            )

        return ActionResult(
            success=True,
            message=f"Clicked '{menu_name}' > '{item_name}'",
            details={"menu": menu_name, "item": item_name}
        )

    def type_in_field(self, field_description: str, text: str, verify_after: Optional[str] = None) -> ActionResult:
        """Click a field and type text into it.

        Args:
            field_description: Description of the field to type in.
            text: The text to type.
            verify_after: Optional verification after typing.

        Returns:
            ActionResult with success status.
        """
        # Find and click the field
        click_result = self.click_element(field_description)

        if not click_result.success:
            return ActionResult(
                success=False,
                message=f"Could not find field: {field_description}",
                retry_suggested=True
            )

        # Wait for field to be focused
        time.sleep(0.2)

        # Type the text
        self.control.type_text(text)

        # Verify if requested
        if verify_after:
            time.sleep(0.3)
            if self.verify_action(verify_after):
                return ActionResult(
                    success=True,
                    message=f"Typed '{text}' in '{field_description}'",
                    details={"text": text, "field": field_description}
                )
            else:
                return ActionResult(
                    success=False,
                    message=f"Typed text but verification failed",
                    retry_suggested=True
                )

        return ActionResult(
            success=True,
            message=f"Typed '{text}' in '{field_description}'",
            details={"text": text, "field": field_description}
        )

    def execute_task(self, task_description: str, verify: bool = False) -> ActionResult:
        """Execute a high-level task using vision guidance.

        OPTIMIZED: Single vision call provides coordinates directly.
        No redundant find_element calls during execution.

        Args:
            task_description: What the user wants to do (e.g., "open a new Chrome window")
            verify: Whether to verify the result (adds latency)

        Returns:
            ActionResult with success status and details.
        """
        result = self._capture_screen_b64()
        if not result:
            return ActionResult(
                success=False,
                message="Could not capture screen",
                retry_suggested=True
            )

        image_b64, (width, height) = result

        # OPTIMIZED PROMPT: Get coordinates directly, no separate find_element calls needed
        prompt = f"""I need to: "{task_description}"

Screen size: {width}x{height} pixels

Look at this screenshot and plan the steps. For EACH click action, provide the EXACT x,y coordinates.

Respond with a JSON object:
{{
    "current_state": "brief description",
    "can_proceed": true/false,
    "steps": [
        {{"action": "click", "x": 500, "y": 300, "description": "clicking the File menu"}},
        {{"action": "type", "text": "text to type"}},
        {{"action": "hotkey", "keys": ["command", "n"]}},
        {{"action": "wait", "seconds": 0.3}}
    ],
    "expected_result": "what should happen"
}}

CRITICAL:
- For EVERY click, provide exact x,y pixel coordinates of the element's CENTER
- Look carefully at the screenshot to find precise positions
- Prefer keyboard shortcuts (hotkey) over clicks when possible - they're faster and more reliable
- Keep steps minimal - don't over-complicate

Respond with ONLY the JSON."""

        response = self._ask_claude_vision(image_b64, prompt)

        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            plan = json.loads(response)
        except json.JSONDecodeError:
            return ActionResult(
                success=False,
                message=f"Failed to plan task: {response[:200]}",
                retry_suggested=True
            )

        if not plan.get("can_proceed", False):
            return ActionResult(
                success=False,
                message=f"Cannot proceed: {plan.get('current_state', 'Unknown state')}",
                details=plan,
                retry_suggested=True
            )

        # Calculate scale factors for coordinate scaling
        screen_width = self.control.screen_width
        screen_height = self.control.screen_height
        if screen_width > SCREENSHOT_MAX_WIDTH:
            scale_x = screen_width / width
            scale_y = screen_height / height
        else:
            scale_x = 1.0
            scale_y = 1.0

        # Execute each step - FAST: use coordinates directly from plan (with scaling)
        steps = plan.get("steps", [])
        for i, step in enumerate(steps):
            action = step.get("action")
            desc = step.get("description", "")
            print(f"[ActionExecutor] Step {i+1}/{len(steps)}: {action} {desc}")

            try:
                if action == "click":
                    # OPTIMIZED: Use coordinates directly from plan (with scaling)
                    x, y = step.get("x", 0), step.get("y", 0)
                    if x > 0 and y > 0:
                        scaled_x = int(x * scale_x)
                        scaled_y = int(y * scale_y)
                        print(f"[ActionExecutor] Click at ({x}, {y}) -> scaled to ({scaled_x}, {scaled_y})")
                        self.control.click(scaled_x, scaled_y)
                    else:
                        print(f"[ActionExecutor] Skipping click - no coordinates provided")

                elif action == "double_click":
                    x, y = step.get("x", 0), step.get("y", 0)
                    if x > 0 and y > 0:
                        scaled_x = int(x * scale_x)
                        scaled_y = int(y * scale_y)
                        self.control.double_click(scaled_x, scaled_y)

                elif action == "right_click":
                    x, y = step.get("x", 0), step.get("y", 0)
                    if x > 0 and y > 0:
                        scaled_x = int(x * scale_x)
                        scaled_y = int(y * scale_y)
                        self.control.right_click(scaled_x, scaled_y)

                elif action == "type":
                    self.control.type_text(step.get("text", ""))

                elif action == "hotkey":
                    keys = step.get("keys", [])
                    if isinstance(keys, list):
                        self.control.hotkey(*keys)
                    else:
                        print(f"[ActionExecutor] Invalid hotkey format: {keys}")

                elif action == "press_key":
                    self.control.press_key(step.get("key", ""))

                elif action == "wait":
                    time.sleep(step.get("seconds", 0.3))

                elif action == "scroll":
                    self.control.scroll(step.get("amount", 0))

                # Minimal delay between actions (UI needs time to respond)
                time.sleep(0.15)

            except Exception as e:
                return ActionResult(
                    success=False,
                    message=f"Step {i+1} failed: {e}",
                    details={"failed_step": step},
                    retry_suggested=True
                )

        # Only verify if explicitly requested (saves 1-2 seconds)
        if verify:
            expected = plan.get("expected_result", "")
            if expected:
                time.sleep(0.3)
                if self.verify_action(expected):
                    return ActionResult(
                        success=True,
                        message=f"Task completed and verified: {task_description}",
                        details={"steps_executed": len(steps), "verification": "passed"}
                    )
                else:
                    return ActionResult(
                        success=False,
                        message=f"Verification failed. Expected: {expected}",
                        details={"steps_executed": len(steps), "verification": "failed"},
                        retry_suggested=True
                    )

        return ActionResult(
            success=True,
            message=f"Completed {len(steps)} steps for: {task_description}",
            details={"steps_executed": len(steps)}
        )


# Singleton instance
_executor: Optional[VisionActionExecutor] = None


def get_action_executor(control: Optional[ComputerControl] = None) -> VisionActionExecutor:
    """Get or create the singleton VisionActionExecutor."""
    global _executor
    if _executor is None:
        if control is None:
            control = ComputerControl()
        _executor = VisionActionExecutor(control)
    return _executor
