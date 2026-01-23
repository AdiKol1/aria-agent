"""
UI Actions via Accessibility API and pyautogui.

This module provides classes for performing actions on UI elements,
including clicking, typing, scrolling, and keyboard shortcuts.
"""
from typing import Optional, List, Union, Tuple
import time
import subprocess

# Try to import pyautogui, but don't fail if not available
try:
    import pyautogui
    pyautogui.FAILSAFE = True  # Move mouse to corner to abort
    pyautogui.PAUSE = 0.1  # Small pause between actions
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    print("[Accessibility] pyautogui not available - some actions will be limited")

from .elements import UIElement, run_applescript, get_finder


class UIActions:
    """Perform actions on UI elements.

    This class provides methods for interacting with UI elements through
    mouse clicks, keyboard input, and other interactions.

    Example:
        actions = UIActions()
        finder = ElementFinder()

        button = finder.find_by_name("Submit")
        if button:
            actions.click(button)
    """

    def __init__(self, safe_mode: bool = True):
        """Initialize UI actions.

        Args:
            safe_mode: If True, add safety checks and delays
        """
        self.safe_mode = safe_mode
        self._last_action_time = 0
        self._min_action_interval = 0.1  # Minimum time between actions

    def _check_pyautogui(self) -> bool:
        """Check if pyautogui is available."""
        if not PYAUTOGUI_AVAILABLE:
            print("[Accessibility] pyautogui not available")
            return False
        return True

    def _safe_delay(self) -> None:
        """Add a safety delay between actions if needed."""
        if self.safe_mode:
            elapsed = time.time() - self._last_action_time
            if elapsed < self._min_action_interval:
                time.sleep(self._min_action_interval - elapsed)
        self._last_action_time = time.time()

    def click(self, target: Union[UIElement, Tuple[int, int]],
              button: str = 'left') -> bool:
        """Click on an element or coordinates.

        Args:
            target: UIElement or (x, y) tuple to click
            button: Mouse button ('left', 'right', or 'middle')

        Returns:
            True if click was successful
        """
        if not self._check_pyautogui():
            return False

        try:
            if isinstance(target, UIElement):
                x, y = target.center
            else:
                x, y = target

            self._safe_delay()
            pyautogui.click(x, y, button=button)
            print(f"[Accessibility] Clicked at ({x}, {y})")
            return True
        except Exception as e:
            print(f"[Accessibility] Click failed: {e}")
            return False

    def double_click(self, target: Union[UIElement, Tuple[int, int]]) -> bool:
        """Double-click on an element or coordinates.

        Args:
            target: UIElement or (x, y) tuple to double-click

        Returns:
            True if double-click was successful
        """
        if not self._check_pyautogui():
            return False

        try:
            if isinstance(target, UIElement):
                x, y = target.center
            else:
                x, y = target

            self._safe_delay()
            pyautogui.doubleClick(x, y)
            print(f"[Accessibility] Double-clicked at ({x}, {y})")
            return True
        except Exception as e:
            print(f"[Accessibility] Double-click failed: {e}")
            return False

    def right_click(self, target: Union[UIElement, Tuple[int, int]]) -> bool:
        """Right-click on an element or coordinates.

        Args:
            target: UIElement or (x, y) tuple to right-click

        Returns:
            True if right-click was successful
        """
        return self.click(target, button='right')

    def triple_click(self, target: Union[UIElement, Tuple[int, int]]) -> bool:
        """Triple-click on an element (useful for selecting lines/paragraphs).

        Args:
            target: UIElement or (x, y) tuple to triple-click

        Returns:
            True if triple-click was successful
        """
        if not self._check_pyautogui():
            return False

        try:
            if isinstance(target, UIElement):
                x, y = target.center
            else:
                x, y = target

            self._safe_delay()
            pyautogui.tripleClick(x, y)
            print(f"[Accessibility] Triple-clicked at ({x}, {y})")
            return True
        except Exception as e:
            print(f"[Accessibility] Triple-click failed: {e}")
            return False

    def move_to(self, target: Union[UIElement, Tuple[int, int]],
                duration: float = 0) -> bool:
        """Move the mouse to an element or coordinates.

        Args:
            target: UIElement or (x, y) tuple to move to
            duration: Time in seconds for the movement (0 = instant)

        Returns:
            True if movement was successful
        """
        if not self._check_pyautogui():
            return False

        try:
            if isinstance(target, UIElement):
                x, y = target.center
            else:
                x, y = target

            self._safe_delay()
            pyautogui.moveTo(x, y, duration=duration)
            return True
        except Exception as e:
            print(f"[Accessibility] Move failed: {e}")
            return False

    def drag_to(self, start: Union[UIElement, Tuple[int, int]],
                end: Union[UIElement, Tuple[int, int]],
                duration: float = 0.5) -> bool:
        """Drag from start to end position.

        Args:
            start: Starting UIElement or (x, y) coordinates
            end: Ending UIElement or (x, y) coordinates
            duration: Time in seconds for the drag operation

        Returns:
            True if drag was successful
        """
        if not self._check_pyautogui():
            return False

        try:
            if isinstance(start, UIElement):
                start_x, start_y = start.center
            else:
                start_x, start_y = start

            if isinstance(end, UIElement):
                end_x, end_y = end.center
            else:
                end_x, end_y = end

            self._safe_delay()
            pyautogui.moveTo(start_x, start_y)
            pyautogui.drag(end_x - start_x, end_y - start_y, duration=duration)
            print(f"[Accessibility] Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})")
            return True
        except Exception as e:
            print(f"[Accessibility] Drag failed: {e}")
            return False

    def type_text(self, text: str, element: UIElement = None,
                  interval: float = 0.0, use_clipboard: bool = True) -> bool:
        """Type text, optionally into a specific element.

        Args:
            text: Text to type
            element: Optional element to click first
            interval: Time between keystrokes (0 = fast)
            use_clipboard: If True, use clipboard for reliability

        Returns:
            True if typing was successful
        """
        if not self._check_pyautogui():
            return False

        try:
            # Click on element first if specified
            if element:
                self.click(element)
                time.sleep(0.1)

            self._safe_delay()

            if use_clipboard:
                # Use clipboard for more reliable typing
                # (handles special characters better)
                import pyperclip
                old_clipboard = pyperclip.paste()
                pyperclip.copy(text)
                pyautogui.hotkey('command', 'v')
                time.sleep(0.1)
                pyperclip.copy(old_clipboard)  # Restore clipboard
            else:
                pyautogui.write(text, interval=interval)

            print(f"[Accessibility] Typed: {text[:50]}{'...' if len(text) > 50 else ''}")
            return True
        except ImportError:
            # pyperclip not available, fall back to direct typing
            try:
                pyautogui.write(text, interval=interval)
                return True
            except Exception as e:
                print(f"[Accessibility] Type failed: {e}")
                return False
        except Exception as e:
            print(f"[Accessibility] Type failed: {e}")
            return False

    def press_key(self, key: str) -> bool:
        """Press a single key.

        Args:
            key: Key name (enter, tab, escape, space, delete, backspace,
                 up, down, left, right, etc.)

        Returns:
            True if key press was successful
        """
        if not self._check_pyautogui():
            return False

        # Map common key names
        key_map = {
            'enter': 'return',
            'return': 'return',
            'esc': 'escape',
            'escape': 'escape',
            'del': 'delete',
            'delete': 'delete',
            'backspace': 'backspace',
            'tab': 'tab',
            'space': 'space',
            'up': 'up',
            'down': 'down',
            'left': 'left',
            'right': 'right',
            'home': 'home',
            'end': 'end',
            'pageup': 'pageup',
            'pagedown': 'pagedown',
            'f1': 'f1', 'f2': 'f2', 'f3': 'f3', 'f4': 'f4',
            'f5': 'f5', 'f6': 'f6', 'f7': 'f7', 'f8': 'f8',
            'f9': 'f9', 'f10': 'f10', 'f11': 'f11', 'f12': 'f12',
        }

        mapped_key = key_map.get(key.lower(), key)

        try:
            self._safe_delay()
            pyautogui.press(mapped_key)
            print(f"[Accessibility] Pressed: {key}")
            return True
        except Exception as e:
            print(f"[Accessibility] Key press failed: {e}")
            return False

    def hotkey(self, *keys: str) -> bool:
        """Press a key combination (hotkey/shortcut).

        Args:
            *keys: Keys to press together (e.g., 'command', 'c' for Cmd+C)

        Returns:
            True if hotkey was successful
        """
        if not self._check_pyautogui():
            return False

        # Map modifier key names
        key_map = {
            'cmd': 'command',
            'command': 'command',
            'ctrl': 'ctrl',
            'control': 'ctrl',
            'alt': 'option',
            'option': 'option',
            'shift': 'shift',
            'win': 'command',  # Map Windows key to Command on Mac
        }

        mapped_keys = [key_map.get(k.lower(), k.lower()) for k in keys]

        try:
            self._safe_delay()
            pyautogui.hotkey(*mapped_keys)
            print(f"[Accessibility] Hotkey: {'+'.join(keys)}")
            return True
        except Exception as e:
            print(f"[Accessibility] Hotkey failed: {e}")
            return False

    def scroll(self, amount: int, direction: str = None) -> bool:
        """Scroll up or down.

        Args:
            amount: Number of "clicks" to scroll. Positive = up, negative = down.
                   If direction is specified, amount is always treated as positive.
            direction: Optional explicit direction ('up' or 'down')

        Returns:
            True if scroll was successful
        """
        if not self._check_pyautogui():
            return False

        try:
            if direction:
                direction = direction.lower()
                amount = abs(amount)
                if direction == 'down':
                    amount = -amount

            self._safe_delay()
            pyautogui.scroll(amount)
            print(f"[Accessibility] Scrolled {'up' if amount > 0 else 'down'} by {abs(amount)}")
            return True
        except Exception as e:
            print(f"[Accessibility] Scroll failed: {e}")
            return False

    def scroll_at(self, target: Union[UIElement, Tuple[int, int]],
                  amount: int) -> bool:
        """Scroll at a specific location.

        Args:
            target: UIElement or (x, y) coordinates to scroll at
            amount: Number of "clicks" to scroll (positive = up, negative = down)

        Returns:
            True if scroll was successful
        """
        if not self._check_pyautogui():
            return False

        try:
            if isinstance(target, UIElement):
                x, y = target.center
            else:
                x, y = target

            self._safe_delay()
            pyautogui.scroll(amount, x=x, y=y)
            return True
        except Exception as e:
            print(f"[Accessibility] Scroll at failed: {e}")
            return False

    def focus(self, element: UIElement) -> bool:
        """Focus on an element by clicking it.

        For text fields, this also selects all text.

        Args:
            element: Element to focus

        Returns:
            True if focus was successful
        """
        if self.click(element):
            # If it's a text field, select all
            if 'text' in element.role.lower() or 'field' in element.role.lower():
                time.sleep(0.1)
                self.hotkey('command', 'a')
            return True
        return False

    def activate_app(self, app_name: str) -> bool:
        """Bring an application to the foreground.

        Args:
            app_name: Name of the application to activate

        Returns:
            True if activation was successful
        """
        script = f'''
        tell application "{app_name}"
            activate
        end tell
        '''
        result = run_applescript(script)
        # Script returns empty on success
        return True

    def open_app(self, app_name: str) -> bool:
        """Open an application.

        Args:
            app_name: Name of the application to open

        Returns:
            True if opening was successful
        """
        script = f'''
        tell application "{app_name}"
            activate
        end tell
        '''
        run_applescript(script)
        time.sleep(0.5)  # Wait for app to open
        return True

    def close_window(self, app: str = None) -> bool:
        """Close the current window.

        Args:
            app: Application name (defaults to frontmost app)

        Returns:
            True if close was successful
        """
        if not app:
            finder = get_finder()
            app = finder.get_active_app()

        if not app:
            return False

        script = f'''
        tell application "{app}"
            close front window
        end tell
        '''
        run_applescript(script)
        return True

    def minimize_window(self, app: str = None) -> bool:
        """Minimize the current window.

        Args:
            app: Application name (defaults to frontmost app)

        Returns:
            True if minimize was successful
        """
        if not app:
            finder = get_finder()
            app = finder.get_active_app()

        if not app:
            return False

        script = f'''
        tell application "{app}"
            set miniaturized of front window to true
        end tell
        '''
        run_applescript(script)
        return True

    def maximize_window(self, app: str = None) -> bool:
        """Maximize/zoom the current window.

        Args:
            app: Application name (defaults to frontmost app)

        Returns:
            True if maximize was successful
        """
        if not app:
            finder = get_finder()
            app = finder.get_active_app()

        if not app:
            return False

        # Use keyboard shortcut for full screen
        self.activate_app(app)
        time.sleep(0.1)
        return self.hotkey('ctrl', 'command', 'f')

    def click_accessibility(self, element_description: str,
                           app: str = None) -> bool:
        """Click on an element using Accessibility API directly.

        This uses AppleScript to click elements by their accessibility properties,
        which can be more reliable than coordinate-based clicking.

        Args:
            element_description: Description of the element (e.g., "button Submit")
            app: Application name (defaults to frontmost app)

        Returns:
            True if click was successful
        """
        if not app:
            finder = get_finder()
            app = finder.get_active_app()

        if not app:
            return False

        # Parse element description
        parts = element_description.lower().split()
        role = parts[0] if parts else "button"
        name = ' '.join(parts[1:]) if len(parts) > 1 else ""

        if name:
            script = f'''
            tell application "System Events"
                tell process "{app}"
                    try
                        click {role} "{name}" of window 1
                        return "success"
                    on error
                        return "error"
                    end try
                end tell
            end tell
            '''
        else:
            script = f'''
            tell application "System Events"
                tell process "{app}"
                    try
                        click {role} 1 of window 1
                        return "success"
                    on error
                        return "error"
                    end try
                end tell
            end tell
            '''

        result = run_applescript(script)
        success = result == "success"
        if success:
            print(f"[Accessibility] Clicked {element_description} via Accessibility API")
        return success


# Singleton instance for convenience
_actions: Optional[UIActions] = None

def get_actions() -> UIActions:
    """Get the shared UIActions instance."""
    global _actions
    if _actions is None:
        _actions = UIActions()
    return _actions
