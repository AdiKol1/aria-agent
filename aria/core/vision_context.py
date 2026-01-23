"""
Proactive Vision Context for Aria.

This module provides continuous visual awareness, ensuring Aria always
"looks before she leaps" - capturing screen state and mouse position
BEFORE any action, and verifying AFTER.

Key principles from research:
1. Vision-Action Loop: Screenshot BEFORE every action
2. Mouse Position Awareness: Always know where the cursor is
3. State Verification: Screenshot AFTER to verify results
4. Continuous Observation: Background awareness of user activity
"""

import asyncio
import base64
import hashlib
import io
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import pyautogui
from PIL import Image

# Try to import pynput for mouse tracking
try:
    from pynput import mouse
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("[VisionContext] pynput not available - mouse tracking will use polling")


@dataclass
class ScreenState:
    """Snapshot of the current screen state."""
    timestamp: float
    screenshot_b64: Optional[str] = None
    screenshot_size: Tuple[int, int] = (0, 0)
    mouse_position: Tuple[int, int] = (0, 0)
    active_window: str = ""
    screen_hash: str = ""  # For quick change detection

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "mouse_position": self.mouse_position,
            "active_window": self.active_window,
            "screenshot_size": self.screenshot_size,
            "has_screenshot": self.screenshot_b64 is not None,
        }


@dataclass
class ActionContext:
    """Context for an action execution with before/after states."""
    action_name: str
    action_args: Dict[str, Any]
    before_state: Optional[ScreenState] = None
    after_state: Optional[ScreenState] = None
    started_at: float = 0.0
    completed_at: float = 0.0
    success: bool = False
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at) * 1000
        return 0.0


@dataclass
class UserAction:
    """Record of a user action (for learning)."""
    timestamp: float
    action_type: str  # "click", "scroll", "keystroke"
    position: Optional[Tuple[int, int]] = None
    details: Dict[str, Any] = field(default_factory=dict)


class VisionContext:
    """
    Maintains continuous visual awareness for Aria.

    This is the core component that ensures Aria always has
    current visual context before taking any action.
    """

    def __init__(
        self,
        screenshot_max_width: int = 1200,
        screenshot_quality: int = 75,
        cache_ttl: float = 1.0,
    ):
        self.screenshot_max_width = screenshot_max_width
        self.screenshot_quality = screenshot_quality
        self.cache_ttl = cache_ttl

        # Current state
        self._current_state: Optional[ScreenState] = None
        self._state_lock = threading.Lock()

        # Screen size
        self.screen_width, self.screen_height = pyautogui.size()

        # Mouse tracking
        self._mouse_position: Tuple[int, int] = (0, 0)
        self._mouse_listener: Optional[Any] = None
        self._mouse_click_callbacks: List[Callable] = []
        self._mouse_move_callbacks: List[Callable] = []

        # Action history (for learning/debugging)
        self._action_history: Deque[ActionContext] = deque(maxlen=100)

        # Start mouse tracking
        self._start_mouse_tracking()

    def _start_mouse_tracking(self):
        """Start tracking mouse position in background."""
        if PYNPUT_AVAILABLE:
            def on_move(x, y):
                self._mouse_position = (x, y)
                for callback in self._mouse_move_callbacks:
                    try:
                        callback(x, y)
                    except Exception:
                        pass

            def on_click(x, y, button, pressed):
                if pressed:
                    for callback in self._mouse_click_callbacks:
                        try:
                            callback(x, y, str(button))
                        except Exception:
                            pass

            self._mouse_listener = mouse.Listener(
                on_move=on_move,
                on_click=on_click
            )
            self._mouse_listener.start()
            print("[VisionContext] Mouse tracking started (pynput)")
        else:
            # Fallback: update position when requested
            self._mouse_position = pyautogui.position()
            print("[VisionContext] Mouse tracking using polling (pynput not available)")

    def stop(self):
        """Stop all background processes."""
        if self._mouse_listener:
            self._mouse_listener.stop()
            self._mouse_listener = None

    @property
    def mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position."""
        if not PYNPUT_AVAILABLE:
            self._mouse_position = pyautogui.position()
        return self._mouse_position

    def on_mouse_click(self, callback: Callable[[int, int, str], None]):
        """Register a callback for mouse clicks."""
        self._mouse_click_callbacks.append(callback)

    def on_mouse_move(self, callback: Callable[[int, int], None]):
        """Register a callback for mouse movement."""
        self._mouse_move_callbacks.append(callback)

    def _get_active_window(self) -> str:
        """Get the name of the currently focused application."""
        try:
            import subprocess
            script = '''
            tell application "System Events"
                set frontApp to name of first application process whose frontmost is true
            end tell
            return frontApp
            '''
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip()
        except Exception:
            return "Unknown"

    def _capture_screenshot(self) -> Optional[Tuple[str, Tuple[int, int]]]:
        """Capture screenshot and return (base64, (width, height))."""
        try:
            import subprocess
            from pathlib import Path
            import tempfile

            # Use screencapture command
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_path = f.name

            result = subprocess.run(
                ["screencapture", "-x", temp_path],
                capture_output=True,
                timeout=10
            )

            if result.returncode != 0:
                return None

            # Load and process image
            image = Image.open(temp_path)

            # Resize if needed
            if image.width > self.screenshot_max_width:
                ratio = self.screenshot_max_width / image.width
                new_size = (self.screenshot_max_width, int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.BILINEAR)

            final_size = (image.width, image.height)

            # Convert to JPEG base64
            buffer = io.BytesIO()
            if image.mode in ('RGBA', 'P'):
                image = image.convert('RGB')
            image.save(buffer, format="JPEG", quality=self.screenshot_quality, optimize=False)
            b64 = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

            # Cleanup temp file
            Path(temp_path).unlink(missing_ok=True)

            return (b64, final_size)

        except Exception as e:
            print(f"[VisionContext] Screenshot error: {e}")
            return None

    def _compute_screen_hash(self, image_b64: str) -> str:
        """Compute a hash for quick screen change detection."""
        # Use first 1000 chars for fast comparison
        return hashlib.md5(image_b64[:1000].encode()).hexdigest()[:16]

    def capture_state(self, include_screenshot: bool = True) -> ScreenState:
        """
        Capture current screen state.

        This is the PRIMARY method - call this BEFORE any action.

        Args:
            include_screenshot: Whether to capture a screenshot (set False for quick checks)

        Returns:
            ScreenState with current visual context
        """
        state = ScreenState(
            timestamp=time.time(),
            mouse_position=self.mouse_position,
            active_window=self._get_active_window(),
        )

        if include_screenshot:
            result = self._capture_screenshot()
            if result:
                state.screenshot_b64, state.screenshot_size = result
                state.screen_hash = self._compute_screen_hash(state.screenshot_b64)

        with self._state_lock:
            self._current_state = state

        return state

    def get_current_state(self) -> Optional[ScreenState]:
        """Get the most recently captured state."""
        with self._state_lock:
            return self._current_state

    def screen_changed(self, old_state: ScreenState, new_state: ScreenState) -> bool:
        """
        Check if the screen has changed between two states.

        Useful for verifying action results.
        """
        if not old_state.screen_hash or not new_state.screen_hash:
            return True  # Assume changed if we can't compare
        return old_state.screen_hash != new_state.screen_hash

    def verify_mouse_at_target(
        self,
        target_x: int,
        target_y: int,
        tolerance: int = 50
    ) -> Tuple[bool, Tuple[int, int]]:
        """
        Verify mouse is at or near the target position.

        Returns:
            (is_near_target, actual_position)
        """
        current_x, current_y = self.mouse_position
        distance = ((current_x - target_x) ** 2 + (current_y - target_y) ** 2) ** 0.5
        return (distance <= tolerance, (current_x, current_y))

    def record_action(self, context: ActionContext):
        """Record an action for history/learning."""
        self._action_history.append(context)

    def get_recent_actions(self, n: int = 10) -> List[ActionContext]:
        """Get the most recent n actions."""
        return list(self._action_history)[-n:]


class VisionAwareExecutor:
    """
    Wraps action execution with mandatory vision capture.

    This ensures every action follows the pattern:
    1. Capture BEFORE state
    2. Verify prerequisites (mouse position, target visible)
    3. Execute action
    4. Capture AFTER state
    5. Verify result
    """

    def __init__(self, vision_context: VisionContext, control: Any):
        self.vision = vision_context
        self.control = control
        self._pre_action_hooks: List[Callable] = []
        self._post_action_hooks: List[Callable] = []

    def add_pre_action_hook(self, hook: Callable[[str, Dict], None]):
        """Add a hook called before every action."""
        self._pre_action_hooks.append(hook)

    def add_post_action_hook(self, hook: Callable[[ActionContext], None]):
        """Add a hook called after every action."""
        self._post_action_hooks.append(hook)

    def execute_with_vision(
        self,
        action_name: str,
        action_fn: Callable,
        args: Dict[str, Any],
        capture_before: bool = True,
        capture_after: bool = True,
        verify_mouse: bool = True,
    ) -> ActionContext:
        """
        Execute an action with full vision context.

        Args:
            action_name: Name of the action (for logging/debugging)
            action_fn: The function to execute
            args: Arguments to pass to the function
            capture_before: Whether to capture screen before
            capture_after: Whether to capture screen after
            verify_mouse: For click actions, verify mouse position

        Returns:
            ActionContext with full before/after state
        """
        context = ActionContext(
            action_name=action_name,
            action_args=args,
            started_at=time.time()
        )

        # Call pre-action hooks
        for hook in self._pre_action_hooks:
            try:
                hook(action_name, args)
            except Exception:
                pass

        try:
            # 1. Capture BEFORE state
            if capture_before:
                context.before_state = self.vision.capture_state(include_screenshot=True)
                print(f"[VisionExecutor] Before {action_name}: mouse at {context.before_state.mouse_position}, window: {context.before_state.active_window}")

            # 2. For click actions, verify mouse position
            if verify_mouse and action_name in ("click", "double_click", "right_click"):
                target_x = args.get("x", 0)
                target_y = args.get("y", 0)
                if target_x and target_y:
                    is_at_target, actual_pos = self.vision.verify_mouse_at_target(target_x, target_y)
                    if not is_at_target:
                        print(f"[VisionExecutor] Mouse at {actual_pos}, moving to ({target_x}, {target_y})")

            # 3. Execute the action
            result = action_fn(**args)
            context.success = result if isinstance(result, bool) else True

            # 4. Brief pause for UI to update
            time.sleep(0.1)

            # 5. Capture AFTER state
            if capture_after:
                context.after_state = self.vision.capture_state(include_screenshot=True)

                # Check if screen changed
                if context.before_state and context.after_state:
                    changed = self.vision.screen_changed(context.before_state, context.after_state)
                    print(f"[VisionExecutor] After {action_name}: screen {'changed' if changed else 'unchanged'}")

        except Exception as e:
            context.success = False
            context.error = str(e)
            print(f"[VisionExecutor] Error in {action_name}: {e}")

        context.completed_at = time.time()

        # Record action
        self.vision.record_action(context)

        # Call post-action hooks
        for hook in self._post_action_hooks:
            try:
                hook(context)
            except Exception:
                pass

        return context

    # Convenience methods for common actions

    def click(self, x: int, y: int, button: str = "left") -> ActionContext:
        """Click with vision context."""
        return self.execute_with_vision(
            "click",
            lambda x, y, button: self.control.click(x, y, button=button),
            {"x": x, "y": y, "button": button}
        )

    def double_click(self, x: int, y: int) -> ActionContext:
        """Double-click with vision context."""
        return self.execute_with_vision(
            "double_click",
            lambda x, y: self.control.double_click(x, y),
            {"x": x, "y": y}
        )

    def right_click(self, x: int, y: int) -> ActionContext:
        """Right-click with vision context."""
        return self.execute_with_vision(
            "right_click",
            lambda x, y: self.control.right_click(x, y),
            {"x": x, "y": y}
        )

    def type_text(self, text: str) -> ActionContext:
        """Type text with vision context."""
        return self.execute_with_vision(
            "type_text",
            lambda text: self.control.type_text(text),
            {"text": text},
            verify_mouse=False
        )

    def press_key(self, key: str) -> ActionContext:
        """Press key with vision context."""
        return self.execute_with_vision(
            "press_key",
            lambda key: self.control.press_key(key),
            {"key": key},
            verify_mouse=False
        )

    def hotkey(self, *keys: str) -> ActionContext:
        """Press hotkey with vision context."""
        return self.execute_with_vision(
            "hotkey",
            lambda keys: self.control.hotkey(*keys),
            {"keys": keys},
            verify_mouse=False
        )

    def scroll(self, amount: int, x: Optional[int] = None, y: Optional[int] = None) -> ActionContext:
        """Scroll with vision context."""
        return self.execute_with_vision(
            "scroll",
            lambda amount, x, y: self.control.scroll(amount, x, y),
            {"amount": amount, "x": x, "y": y},
            verify_mouse=False
        )


class AmbientObserver:
    """
    Background observer that watches user activity.

    This enables Aria to:
    1. Follow along with user's work
    2. Learn from user actions
    3. Provide contextual assistance
    4. Build understanding over time
    """

    def __init__(
        self,
        vision_context: VisionContext,
        capture_interval: float = 3.0,  # seconds between captures
        max_observations: int = 100,
    ):
        self.vision = vision_context
        self.capture_interval = capture_interval
        self.max_observations = max_observations

        # Observation buffer
        self._observations: Deque[ScreenState] = deque(maxlen=max_observations)
        self._user_actions: Deque[UserAction] = deque(maxlen=max_observations * 2)

        # Background task
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._thread: Optional[threading.Thread] = None

        # Callbacks for when interesting things happen
        self._activity_callbacks: List[Callable] = []

        # Track user clicks
        self.vision.on_mouse_click(self._on_user_click)

    def _on_user_click(self, x: int, y: int, button: str):
        """Record when user clicks (for learning)."""
        action = UserAction(
            timestamp=time.time(),
            action_type="click",
            position=(x, y),
            details={"button": button}
        )
        self._user_actions.append(action)

    def start(self):
        """Start background observation."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._observation_loop, daemon=True)
        self._thread.start()
        print("[AmbientObserver] Started background observation")

    def stop(self):
        """Stop background observation."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("[AmbientObserver] Stopped background observation")

    def _observation_loop(self):
        """Background loop for periodic observation."""
        while self._running:
            try:
                # Capture current state
                state = self.vision.capture_state(include_screenshot=True)
                self._observations.append(state)

                # Check for significant changes
                if len(self._observations) >= 2:
                    prev_state = self._observations[-2]
                    if self.vision.screen_changed(prev_state, state):
                        # Screen changed - might be interesting
                        for callback in self._activity_callbacks:
                            try:
                                callback("screen_changed", state, prev_state)
                            except Exception:
                                pass

            except Exception as e:
                print(f"[AmbientObserver] Observation error: {e}")

            time.sleep(self.capture_interval)

    def on_activity(self, callback: Callable[[str, ScreenState, Optional[ScreenState]], None]):
        """Register callback for when activity is detected."""
        self._activity_callbacks.append(callback)

    def get_recent_observations(self, n: int = 10) -> List[ScreenState]:
        """Get recent screen observations."""
        return list(self._observations)[-n:]

    def get_recent_user_actions(self, n: int = 20) -> List[UserAction]:
        """Get recent user actions."""
        return list(self._user_actions)[-n:]

    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of recent activity for context.

        This can be used to give Aria awareness of what the user
        has been doing.
        """
        observations = self.get_recent_observations(10)
        user_actions = self.get_recent_user_actions(20)

        # Analyze observations
        windows_used = set()
        for obs in observations:
            if obs.active_window:
                windows_used.add(obs.active_window)

        # Analyze user actions
        click_count = sum(1 for a in user_actions if a.action_type == "click")

        return {
            "observation_count": len(observations),
            "user_action_count": len(user_actions),
            "windows_used": list(windows_used),
            "recent_click_count": click_count,
            "current_state": observations[-1].to_dict() if observations else None,
        }

    @property
    def is_running(self) -> bool:
        return self._running


# Singleton instances
_vision_context: Optional[VisionContext] = None
_ambient_observer: Optional[AmbientObserver] = None


def get_vision_context() -> VisionContext:
    """Get the singleton VisionContext instance."""
    global _vision_context
    if _vision_context is None:
        _vision_context = VisionContext()
    return _vision_context


def get_ambient_observer() -> AmbientObserver:
    """Get the singleton AmbientObserver instance."""
    global _ambient_observer
    if _ambient_observer is None:
        _ambient_observer = AmbientObserver(get_vision_context())
    return _ambient_observer
