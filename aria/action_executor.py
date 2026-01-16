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

from .config import ANTHROPIC_API_KEY, CLAUDE_MODEL
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

    def _capture_screen_b64(self) -> Optional[Tuple[str, Tuple[int, int]]]:
        """Capture screen and return (base64, (width, height))."""
        result = self.screen.capture_to_base64_with_size()
        return result

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
            Dict with x, y coordinates and element info, or None if not found.
        """
        result = self._capture_screen_b64()
        if not result:
            return None

        image_b64, (width, height) = result

        prompt = f"""Find the UI element: "{description}"

Screen size: {width}x{height} pixels

If you can find it, respond with ONLY a JSON object like this:
{{"found": true, "x": 500, "y": 300, "element": "File menu button", "confidence": "high"}}

The x,y coordinates should be the CENTER of the element, in screen pixels.
Be precise - look carefully at the actual position of the element.

If you cannot find it, respond with:
{{"found": false, "reason": "why not found", "suggestions": ["alternative approach"]}}

Respond with ONLY the JSON, no other text."""

        response = self._ask_claude_vision(image_b64, prompt)

        try:
            # Extract JSON from response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            result = json.loads(response)
            return result if result.get("found") else None
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

    def execute_task(self, task_description: str) -> ActionResult:
        """Execute a high-level task using vision guidance.

        This method analyzes the screen, plans the steps, and executes them
        with verification.

        Args:
            task_description: What the user wants to do (e.g., "open a new Chrome window")

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

        prompt = f"""I need to: "{task_description}"

Screen size: {width}x{height} pixels

Look at this screenshot and tell me exactly what steps I should take to accomplish this task.
Consider the current state of the screen - what app is open, what's visible, etc.

IMPORTANT: For click actions, provide a clear TARGET DESCRIPTION - this is the most critical part!
The system will use vision to find the element at execution time, so describe what to click clearly.

Respond with a JSON object containing the steps:
{{
    "current_state": "description of what's currently on screen",
    "can_proceed": true/false,
    "steps": [
        {{"action": "click", "target": "clear description like: the plus button in the toolbar, the File menu, the red close button"}},
        {{"action": "type", "text": "text to type"}},
        {{"action": "hotkey", "keys": ["command", "n"]}},
        {{"action": "wait", "seconds": 0.5}}
    ],
    "expected_result": "what should be visible after completing these steps"
}}

CRITICAL FOR CLICKS:
- "target" must be a CLEAR, SPECIFIC description of the UI element
- Examples: "the plus (+) button in the Calendar toolbar", "the File menu in the menu bar"
- The system will use vision to find the exact location, so description quality matters!

If the task cannot be done from the current screen state, explain why and set can_proceed to false.

Respond with ONLY the JSON, no other text."""

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

        # Execute each step
        steps = plan.get("steps", [])
        for i, step in enumerate(steps):
            action = step.get("action")
            print(f"[ActionExecutor] Step {i+1}/{len(steps)}: {action} - {step}")

            try:
                if action == "click":
                    # Use vision-guided clicking with target description
                    target = step.get("target", "")
                    if target:
                        # Re-capture screen and find element with vision
                        element = self.find_element(target)
                        if element:
                            x, y = element["x"], element["y"]
                            print(f"[ActionExecutor] Vision found '{target}' at ({x}, {y})")
                            self.control.click(x, y)
                        else:
                            # Fallback to provided coordinates if vision fails
                            x, y = step.get("x", 0), step.get("y", 0)
                            print(f"[ActionExecutor] Vision failed, using fallback coords ({x}, {y})")
                            if x > 0 and y > 0:
                                self.control.click(x, y)
                            else:
                                print(f"[ActionExecutor] Skipping click - no valid coordinates")
                    else:
                        # No target description, use raw coordinates
                        x, y = step.get("x", 0), step.get("y", 0)
                        self.control.click(x, y)
                elif action == "type":
                    self.control.type_text(step.get("text", ""))
                elif action == "hotkey":
                    keys = step.get("keys", [])
                    self.control.hotkey(keys)
                elif action == "press_key":
                    self.control.press_key(step.get("key", ""))
                elif action == "wait":
                    time.sleep(step.get("seconds", 0.5))
                elif action == "scroll":
                    self.control.scroll(step.get("amount", 0))

                # Small delay between actions
                time.sleep(0.2)

            except Exception as e:
                return ActionResult(
                    success=False,
                    message=f"Step {i+1} failed: {e}",
                    details={"failed_step": step},
                    retry_suggested=True
                )

        # Verify the result
        expected = plan.get("expected_result", "")
        if expected:
            time.sleep(0.5)  # Wait for UI to settle
            if self.verify_action(expected):
                return ActionResult(
                    success=True,
                    message=f"Task completed: {task_description}",
                    details={"steps_executed": len(steps), "verification": "passed"}
                )
            else:
                return ActionResult(
                    success=False,
                    message=f"Task executed but verification failed. Expected: {expected}",
                    details={"steps_executed": len(steps), "verification": "failed"},
                    retry_suggested=True
                )

        return ActionResult(
            success=True,
            message=f"Task completed: {task_description}",
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
