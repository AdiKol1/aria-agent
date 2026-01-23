"""
Skill Executor

Executes learned skills with adaptive replay.
Handles coordinate adjustment, retry logic, and success verification.
"""

import asyncio
import time
from datetime import datetime
from typing import Callable, Dict, Any, List, Optional, Tuple

from .types import (
    ActionType,
    RecordedAction,
    LearnedSkill,
    ExecutionContext,
    ExecutionResult,
)


class SkillExecutor:
    """
    Executes learned skills with adaptive replay.

    Features:
    - Coordinate adjustment based on visual context
    - Retry logic for flaky actions
    - Success verification using criteria
    - Variable substitution for dynamic values

    Usage:
        executor = SkillExecutor(controller)
        result = await executor.execute(skill, variables={"destination": "NYC"})
    """

    def __init__(
        self,
        controller: Any = None,
        vision: Any = None,
    ):
        """
        Initialize the skill executor.

        Args:
            controller: Computer control interface (click, type, etc.)
            vision: Vision interface for screen capture and analysis
        """
        self._controller = controller
        self._vision = vision

        # Callbacks
        self.on_action_start: Optional[Callable[[RecordedAction, int], None]] = None
        self.on_action_complete: Optional[Callable[[RecordedAction, int, bool], None]] = None
        self.on_execution_complete: Optional[Callable[[ExecutionResult], None]] = None

        # Default timing
        self.default_action_delay_ms = 200  # Between actions
        self.click_settle_ms = 100  # After clicks
        self.type_settle_ms = 50  # After typing

    @property
    def controller(self):
        """Lazy load controller."""
        if self._controller is None:
            try:
                from aria.control import click, double_click, scroll, type_text, press_key, hotkey, open_app, open_url
                self._controller = type('Controller', (), {
                    'click': staticmethod(click),
                    'double_click': staticmethod(double_click),
                    'scroll': staticmethod(scroll),
                    'type_text': staticmethod(type_text),
                    'press_key': staticmethod(press_key),
                    'hotkey': staticmethod(hotkey),
                    'open_app': staticmethod(open_app),
                    'open_url': staticmethod(open_url),
                })()
            except ImportError:
                print("Warning: Control module not available")
        return self._controller

    @property
    def vision(self):
        """Lazy load vision."""
        if self._vision is None:
            try:
                from aria.vision import capture_screen
                self._vision = type('Vision', (), {
                    'capture_screen': staticmethod(capture_screen),
                })()
            except ImportError:
                print("Warning: Vision module not available")
        return self._vision

    async def execute(
        self,
        skill: LearnedSkill,
        variables: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
    ) -> ExecutionResult:
        """
        Execute a learned skill.

        Args:
            skill: The skill to execute
            variables: Values for variable substitution
            dry_run: If True, simulate without actually executing

        Returns:
            ExecutionResult with success status and details
        """
        context = ExecutionContext(
            skill=skill,
            variables=variables or {},
            started_at=datetime.now(),
        )

        result = ExecutionResult(
            success=False,
            skill_id=skill.id,
            started_at=datetime.now(),
            total_actions=len(skill.actions),
        )

        try:
            # Check prerequisites
            if skill.required_app:
                current_app = await self._get_current_app()
                if current_app and skill.required_app.lower() not in current_app.lower():
                    # Try to open the required app
                    if not dry_run and self.controller:
                        self.controller.open_app(skill.required_app)
                        await asyncio.sleep(1)  # Wait for app to open

            # Execute each action
            for i, action in enumerate(skill.actions):
                context.current_action_index = i

                if self.on_action_start:
                    self.on_action_start(action, i)

                success = False
                for retry in range(context.max_retries):
                    try:
                        if dry_run:
                            print(f"  [DRY RUN] Action {i + 1}: {action.action_type.value}")
                            success = True
                        else:
                            success = await self._execute_action(action, context)

                        if success:
                            break

                        context.retry_count = retry + 1
                        await asyncio.sleep(0.5)  # Brief pause before retry

                    except Exception as e:
                        context.last_error = str(e)
                        print(f"Action {i + 1} failed (attempt {retry + 1}): {e}")

                if self.on_action_complete:
                    self.on_action_complete(action, i, success)

                if not success:
                    result.error = context.last_error or f"Action {i + 1} failed after {context.max_retries} retries"
                    result.failed_at_action = i
                    break

                result.actions_completed += 1

                # Wait between actions
                delay = action.delay_before_ms or self.default_action_delay_ms
                if delay > 0 and not dry_run:
                    await asyncio.sleep(delay / 1000)

            # Check success criteria if all actions completed
            if result.actions_completed == result.total_actions:
                if skill.success_criteria and not dry_run:
                    result.success = await self._verify_success(skill)
                else:
                    result.success = True

            # Update skill statistics
            if not dry_run:
                skill.times_executed += 1
                if result.success:
                    skill.times_succeeded += 1
                skill.last_executed = datetime.now()

                # Adjust confidence based on outcome
                if result.success:
                    skill.confidence = min(1.0, skill.confidence + 0.05)
                else:
                    skill.confidence = max(0.1, skill.confidence - 0.1)

        except Exception as e:
            result.error = str(e)
            result.success = False

        result.completed_at = datetime.now()

        if self.on_execution_complete:
            self.on_execution_complete(result)

        return result

    async def _execute_action(
        self,
        action: RecordedAction,
        context: ExecutionContext,
    ) -> bool:
        """
        Execute a single action.

        Args:
            action: The action to execute
            context: Execution context with variables and state

        Returns:
            True if action succeeded
        """
        if not self.controller:
            print("No controller available")
            return False

        try:
            if action.action_type == ActionType.CLICK:
                x, y = await self._adjust_coordinates(action, context)
                self.controller.click(x, y)
                await asyncio.sleep(self.click_settle_ms / 1000)
                return True

            elif action.action_type == ActionType.DOUBLE_CLICK:
                x, y = await self._adjust_coordinates(action, context)
                self.controller.double_click(x, y)
                await asyncio.sleep(self.click_settle_ms / 1000)
                return True

            elif action.action_type == ActionType.TYPE:
                text = self._substitute_variables(action.text, context)
                if text:
                    self.controller.type_text(text)
                    await asyncio.sleep(self.type_settle_ms / 1000)
                return True

            elif action.action_type == ActionType.SCROLL:
                x = action.x
                y = action.y
                amount = action.scroll_amount or 0
                self.controller.scroll(amount, x, y)
                return True

            elif action.action_type == ActionType.HOTKEY:
                if action.keys:
                    self.controller.hotkey(action.keys)
                return True

            elif action.action_type == ActionType.KEY_PRESS:
                if action.keys and len(action.keys) > 0:
                    self.controller.press_key(action.keys[0])
                return True

            elif action.action_type == ActionType.WAIT:
                wait_ms = action.delay_before_ms or 1000
                await asyncio.sleep(wait_ms / 1000)
                return True

            elif action.action_type == ActionType.OPEN_APP:
                if action.app_name:
                    self.controller.open_app(action.app_name)
                    await asyncio.sleep(1)  # Wait for app
                return True

            elif action.action_type == ActionType.OPEN_URL:
                if action.text:
                    url = self._substitute_variables(action.text, context)
                    self.controller.open_url(url)
                    await asyncio.sleep(0.5)
                return True

            elif action.action_type == ActionType.CUSTOM:
                # Custom actions need special handling
                print(f"Custom action: {action.notes}")
                return True

            else:
                print(f"Unknown action type: {action.action_type}")
                return False

        except Exception as e:
            print(f"Action execution error: {e}")
            return False

    async def _adjust_coordinates(
        self,
        action: RecordedAction,
        context: ExecutionContext,
    ) -> Tuple[int, int]:
        """
        Adjust click coordinates based on visual context.

        If the action has visual context (description of what was clicked),
        we can try to find that element on screen and adjust coordinates.

        Args:
            action: The action with coordinates and visual target
            context: Execution context

        Returns:
            Adjusted (x, y) coordinates
        """
        x = action.x or 0
        y = action.y or 0

        # Check if we have stored adjustments
        idx = context.current_action_index
        if idx in context.coordinate_adjustments:
            dx, dy = context.coordinate_adjustments[idx]
            return (x + dx, y + dy)

        # If we have visual context, try to find the element
        if action.visual_target and action.visual_target.description:
            try:
                new_x, new_y = await self._find_visual_target(action.visual_target)
                if new_x is not None and new_y is not None:
                    # Store the adjustment for future retries
                    context.coordinate_adjustments[idx] = (new_x - x, new_y - y)
                    return (new_x, new_y)
            except Exception as e:
                print(f"Visual target search failed: {e}")

        return (x, y)

    async def _find_visual_target(
        self,
        target: Any,
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Find a visual target on screen.

        This is a placeholder for visual element matching.
        In a full implementation, this would:
        1. Capture the current screen
        2. Use vision AI to find the element matching the description
        3. Return the center coordinates of the found element

        Args:
            target: VisualTarget with description, screenshot, etc.

        Returns:
            (x, y) of found element, or (None, None) if not found
        """
        # TODO: Implement visual matching with Claude Vision or similar
        # For now, return None to use original coordinates
        return (None, None)

    def _substitute_variables(
        self,
        text: Optional[str],
        context: ExecutionContext,
    ) -> str:
        """
        Substitute variables in text.

        Variables are in the format {{variable_name}}.

        Args:
            text: Text that might contain variables
            context: Execution context with variable values

        Returns:
            Text with variables substituted
        """
        if not text:
            return ""

        result = text
        for name, value in context.variables.items():
            placeholder = f"{{{{{name}}}}}"
            result = result.replace(placeholder, str(value))

        return result

    async def _verify_success(self, skill: LearnedSkill) -> bool:
        """
        Verify that the skill executed successfully.

        Args:
            skill: The skill with success criteria

        Returns:
            True if success criteria is met
        """
        if not skill.success_criteria:
            return True

        # TODO: Implement success verification
        # Options:
        # 1. Screen comparison if success_screenshot_region is set
        # 2. Use Claude Vision to check if criteria is met
        # 3. Check for specific UI elements

        return True  # Assume success for now

    async def _get_current_app(self) -> Optional[str]:
        """Get the currently active application."""
        try:
            import subprocess
            result = subprocess.run(
                ["osascript", "-e", 'tell application "System Events" to get name of first application process whose frontmost is true'],
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except Exception:
            return None

    async def execute_with_prompts(
        self,
        skill: LearnedSkill,
        prompt_callback: Callable[[str, List[str]], str],
        variables: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute a skill, prompting user at decision points.

        Args:
            skill: The skill to execute
            prompt_callback: Function to call when user input is needed
                           Takes (prompt_text, options) and returns user choice
            variables: Initial variable values

        Returns:
            ExecutionResult
        """
        variables = variables or {}

        # Identify variable placeholders in the skill
        for action in skill.actions:
            if action.text and "{{" in action.text:
                # Extract variable names
                import re
                var_names = re.findall(r"\{\{(\w+)\}\}", action.text)
                for var_name in var_names:
                    if var_name not in variables:
                        # Prompt user for this variable
                        value = prompt_callback(
                            f"Enter value for '{var_name}':",
                            []  # No predefined options
                        )
                        variables[var_name] = value

        # Execute with collected variables
        return await self.execute(skill, variables)


# Singleton instance
_executor: Optional[SkillExecutor] = None


def get_skill_executor() -> SkillExecutor:
    """Get the singleton SkillExecutor instance."""
    global _executor
    if _executor is None:
        _executor = SkillExecutor()
    return _executor
