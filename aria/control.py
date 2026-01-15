"""
Computer Control for Aria

Uses pyautogui and AppleScript to control the Mac.
"""

import subprocess
import time
from typing import Optional, List, Tuple

import pyautogui

# Safety settings
pyautogui.FAILSAFE = True  # Move mouse to corner to abort
pyautogui.PAUSE = 0.1  # Small pause between actions


class ComputerControl:
    """Controls the Mac - mouse, keyboard, apps."""

    def __init__(self):
        # Get screen size
        self.screen_width, self.screen_height = pyautogui.size()

    # =========================================================================
    # Mouse Control
    # =========================================================================

    def click(self, x: int, y: int, clicks: int = 1, button: str = "left") -> bool:
        """
        Click at coordinates.

        Args:
            x, y: Screen coordinates
            clicks: Number of clicks (1 for single, 2 for double)
            button: "left", "right", or "middle"
        """
        try:
            pyautogui.click(x, y, clicks=clicks, button=button)
            return True
        except Exception as e:
            print(f"Click error: {e}")
            return False

    def double_click(self, x: int, y: int) -> bool:
        """Double-click at coordinates."""
        return self.click(x, y, clicks=2)

    def right_click(self, x: int, y: int) -> bool:
        """Right-click at coordinates."""
        return self.click(x, y, button="right")

    def move_to(self, x: int, y: int, duration: float = 0.2) -> bool:
        """Move mouse to coordinates smoothly."""
        try:
            pyautogui.moveTo(x, y, duration=duration)
            return True
        except Exception as e:
            print(f"Move error: {e}")
            return False

    def drag_to(
        self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5
    ) -> bool:
        """Drag from start to end coordinates."""
        try:
            pyautogui.moveTo(start_x, start_y)
            pyautogui.drag(end_x - start_x, end_y - start_y, duration=duration)
            return True
        except Exception as e:
            print(f"Drag error: {e}")
            return False

    def scroll(self, amount: int, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """
        Scroll the mouse wheel.

        Args:
            amount: Positive = up, negative = down
            x, y: Optional position to scroll at
        """
        try:
            if x is not None and y is not None:
                pyautogui.moveTo(x, y)
            pyautogui.scroll(amount)
            return True
        except Exception as e:
            print(f"Scroll error: {e}")
            return False

    # =========================================================================
    # Keyboard Control
    # =========================================================================

    def type_text(self, text: str, interval: float = 0.02) -> bool:
        """
        Type text character by character.

        Args:
            text: Text to type
            interval: Delay between characters
        """
        try:
            # Use pyperclip + paste for reliable text input (handles all characters)
            import pyperclip
            pyperclip.copy(text)
            time.sleep(0.1)
            pyautogui.hotkey('command', 'v')
            time.sleep(0.1)
            return True
        except Exception as e:
            # Fall back to typewrite for simple ASCII
            try:
                pyautogui.typewrite(text, interval=interval)
                return True
            except Exception as e2:
                print(f"Type error: {e2}")
                return False

    def press_key(self, key: str) -> bool:
        """
        Press a single key.

        Args:
            key: Key name (e.g., "enter", "tab", "escape", "space")
        """
        try:
            pyautogui.press(key)
            return True
        except Exception as e:
            print(f"Key press error: {e}")
            return False

    def hotkey(self, *keys: str) -> bool:
        """
        Press a keyboard shortcut.

        Args:
            keys: Keys to press together (e.g., "command", "c" for Cmd+C)
        """
        try:
            pyautogui.hotkey(*keys)
            return True
        except Exception as e:
            print(f"Hotkey error: {e}")
            return False

    # Common shortcuts
    def copy(self) -> bool:
        """Copy (Cmd+C)."""
        return self.hotkey("command", "c")

    def paste(self) -> bool:
        """Paste (Cmd+V)."""
        return self.hotkey("command", "v")

    def cut(self) -> bool:
        """Cut (Cmd+X)."""
        return self.hotkey("command", "x")

    def undo(self) -> bool:
        """Undo (Cmd+Z)."""
        return self.hotkey("command", "z")

    def redo(self) -> bool:
        """Redo (Cmd+Shift+Z)."""
        return self.hotkey("command", "shift", "z")

    def select_all(self) -> bool:
        """Select all (Cmd+A)."""
        return self.hotkey("command", "a")

    def save(self) -> bool:
        """Save (Cmd+S)."""
        return self.hotkey("command", "s")

    def new_tab(self) -> bool:
        """New tab (Cmd+T)."""
        return self.hotkey("command", "t")

    def close_tab(self) -> bool:
        """Close tab (Cmd+W)."""
        return self.hotkey("command", "w")

    def switch_app(self) -> bool:
        """Switch app (Cmd+Tab)."""
        return self.hotkey("command", "tab")

    def spotlight(self) -> bool:
        """Open Spotlight (Cmd+Space)."""
        return self.hotkey("command", "space")

    # =========================================================================
    # App Control (AppleScript)
    # =========================================================================

    def run_applescript(self, script: str) -> Optional[str]:
        """Run an AppleScript and return the result."""
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"AppleScript error: {result.stderr}")
                return None
        except Exception as e:
            print(f"AppleScript execution error: {e}")
            return None

    def open_app(self, app_name: str) -> bool:
        """Open an application by name."""
        script = f'tell application "{app_name}" to activate'
        return self.run_applescript(script) is not None

    def quit_app(self, app_name: str) -> bool:
        """Quit an application by name."""
        script = f'tell application "{app_name}" to quit'
        return self.run_applescript(script) is not None

    def get_frontmost_app(self) -> str:
        """Get the name of the frontmost application."""
        script = '''
        tell application "System Events"
            set frontApp to name of first application process whose frontmost is true
        end tell
        return frontApp
        '''
        result = self.run_applescript(script)
        return result if result else "Unknown"

    def open_url(self, url: str) -> bool:
        """Open a URL in the default browser."""
        try:
            subprocess.run(["open", url], check=True)
            return True
        except Exception as e:
            print(f"Open URL error: {e}")
            return False

    def open_file(self, path: str) -> bool:
        """Open a file with its default application."""
        try:
            subprocess.run(["open", path], check=True)
            return True
        except Exception as e:
            print(f"Open file error: {e}")
            return False

    # =========================================================================
    # Window Management
    # =========================================================================

    def get_window_list(self) -> List[dict]:
        """Get list of open windows."""
        script = '''
        tell application "System Events"
            set windowList to {}
            repeat with proc in (every process whose visible is true)
                set procName to name of proc
                repeat with win in (every window of proc)
                    set winName to name of win
                    set end of windowList to procName & ": " & winName
                end repeat
            end repeat
            return windowList
        end tell
        '''
        result = self.run_applescript(script)
        if result:
            return [{"name": w.strip()} for w in result.split(",")]
        return []

    def focus_window(self, app_name: str, window_name: Optional[str] = None) -> bool:
        """Focus a specific window."""
        if window_name:
            script = f'''
            tell application "{app_name}"
                activate
                set index of window "{window_name}" to 1
            end tell
            '''
        else:
            script = f'tell application "{app_name}" to activate'
        return self.run_applescript(script) is not None


# Singleton instance
_control: Optional[ComputerControl] = None


def get_control() -> ComputerControl:
    """Get the singleton ComputerControl instance."""
    global _control
    if _control is None:
        _control = ComputerControl()
    return _control
