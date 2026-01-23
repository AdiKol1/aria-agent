"""
UI Element discovery via macOS Accessibility API.

This module provides classes for discovering and representing UI elements
using the macOS Accessibility API through AppleScript/System Events.
"""
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import subprocess
import time
import re


def run_applescript(script: str, timeout: int = 10) -> str:
    """Run an AppleScript and return the result.

    Args:
        script: AppleScript code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Output from the script, or empty string on error
    """
    try:
        result = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode != 0 and result.stderr:
            # Check for permission errors
            if "not allowed" in result.stderr.lower() or "permission" in result.stderr.lower():
                print(f"[Accessibility] Permission error - enable Accessibility in System Settings")
            else:
                print(f"[Accessibility] AppleScript error: {result.stderr.strip()}")
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print(f"[Accessibility] AppleScript timed out after {timeout}s")
        return ""
    except Exception as e:
        print(f"[Accessibility] AppleScript error: {e}")
        return ""


@dataclass
class UIElement:
    """Represents a UI element from the Accessibility API.

    Attributes:
        name: The element's accessibility name/label
        role: The element's role (button, text field, menu item, etc.)
        position: (x, y) position of the element
        size: (width, height) dimensions
        app: Name of the application containing this element
        is_focused: Whether this element currently has focus
        value: The element's value (for text fields, etc.)
        subrole: More specific role information
        description: Accessibility description
        enabled: Whether the element is enabled/interactive
        children_count: Number of child elements
    """
    name: str
    role: str
    position: Tuple[int, int]
    size: Tuple[int, int]
    app: str
    is_focused: bool = False
    value: Optional[str] = None
    subrole: Optional[str] = None
    description: Optional[str] = None
    enabled: bool = True
    children_count: int = 0

    @property
    def center(self) -> Tuple[int, int]:
        """Get the center point of the element."""
        return (
            self.position[0] + self.size[0] // 2,
            self.position[1] + self.size[1] // 2
        )

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Get bounds as (x, y, width, height)."""
        return (self.position[0], self.position[1], self.size[0], self.size[1])

    @property
    def center_x(self) -> int:
        """Get center X coordinate."""
        return self.center[0]

    @property
    def center_y(self) -> int:
        """Get center Y coordinate."""
        return self.center[1]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'role': self.role,
            'x': self.position[0],
            'y': self.position[1],
            'width': self.size[0],
            'height': self.size[1],
            'center_x': self.center_x,
            'center_y': self.center_y,
            'app': self.app,
            'is_focused': self.is_focused,
            'value': self.value,
            'subrole': self.subrole,
            'description': self.description,
            'enabled': self.enabled,
        }

    def __str__(self) -> str:
        return f"UIElement({self.role}: '{self.name}' at {self.center})"


class ElementFinder:
    """Find UI elements using the macOS Accessibility API.

    This class provides methods to discover UI elements in applications,
    the dock, menu bar, and system-wide.

    Example:
        finder = ElementFinder()
        button = finder.find_by_name("Submit", app="Safari")
        if button:
            print(f"Found button at {button.center}")
    """

    # Cache for expensive operations
    _cache: Dict[str, Tuple[float, Any]] = {}
    _cache_ttl: float = 2.0  # Cache lifetime in seconds

    def __init__(self, cache_ttl: float = 2.0):
        """Initialize the element finder.

        Args:
            cache_ttl: How long to cache element lists (in seconds)
        """
        self._cache_ttl = cache_ttl

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self._cache:
            timestamp, value = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return value
        return None

    def _set_cached(self, key: str, value: Any) -> None:
        """Store value in cache."""
        self._cache[key] = (time.time(), value)

    def clear_cache(self) -> None:
        """Clear the element cache."""
        self._cache.clear()

    def get_active_app(self) -> str:
        """Get the name of the currently active/frontmost application.

        Returns:
            Application name, or empty string if unable to determine
        """
        script = '''
        tell application "System Events"
            set frontApp to first application process whose frontmost is true
            return name of frontApp
        end tell
        '''
        return run_applescript(script)

    def find_by_name(self, name: str, app: str = None,
                     exact_match: bool = False) -> Optional[UIElement]:
        """Find an element by its name/label.

        Args:
            name: Name to search for
            app: Application to search in (defaults to frontmost app)
            exact_match: If True, require exact name match

        Returns:
            UIElement if found, None otherwise
        """
        if not app:
            app = self.get_active_app()
        if not app:
            return None

        # Escape special characters for AppleScript
        name_escaped = name.replace('"', '\\"')

        script = f'''
        tell application "System Events"
            tell process "{app}"
                try
                    set allElements to every UI element of window 1
                    repeat with elem in allElements
                        try
                            set elemName to name of elem
                            if elemName is not missing value then
                                if elemName contains "{name_escaped}" then
                                    set elemPos to position of elem
                                    set elemSize to size of elem
                                    set elemRole to role of elem
                                    set elemFocused to focused of elem
                                    return elemName & "|" & (item 1 of elemPos) & "|" & (item 2 of elemPos) & "|" & (item 1 of elemSize) & "|" & (item 2 of elemSize) & "|" & elemRole & "|" & elemFocused
                                end if
                            end if
                        end try
                    end repeat
                end try
            end tell
        end tell
        return ""
        '''

        result = run_applescript(script)
        if result and '|' in result:
            parts = result.split('|')
            if len(parts) >= 6:
                try:
                    return UIElement(
                        name=parts[0],
                        role=parts[5] if len(parts) > 5 else "unknown",
                        position=(int(parts[1]), int(parts[2])),
                        size=(int(parts[3]), int(parts[4])),
                        app=app,
                        is_focused=parts[6].lower() == 'true' if len(parts) > 6 else False
                    )
                except (ValueError, IndexError):
                    pass

        return None

    def find_by_role(self, role: str, app: str = None,
                     limit: int = 50) -> List[UIElement]:
        """Find all elements with the given role.

        Args:
            role: Role to search for (button, text field, menu item, etc.)
            app: Application to search in (defaults to frontmost app)
            limit: Maximum number of elements to return

        Returns:
            List of matching UIElements
        """
        if not app:
            app = self.get_active_app()
        if not app:
            return []

        # Map common role names to AppleScript role descriptions
        role_map = {
            'button': 'button',
            'textfield': 'text field',
            'text field': 'text field',
            'checkbox': 'checkbox',
            'radio': 'radio button',
            'radiobutton': 'radio button',
            'menu': 'menu',
            'menuitem': 'menu item',
            'menu item': 'menu item',
            'list': 'list',
            'table': 'table',
            'image': 'image',
            'statictext': 'static text',
            'static text': 'static text',
            'link': 'link',
            'group': 'group',
            'toolbar': 'toolbar',
            'scrollarea': 'scroll area',
            'scroll area': 'scroll area',
            'popupbutton': 'pop up button',
            'popup': 'pop up button',
        }

        mapped_role = role_map.get(role.lower(), role)

        script = f'''
        tell application "System Events"
            tell process "{app}"
                try
                    set output to ""
                    set elementCount to 0
                    set allElements to every {mapped_role} of window 1
                    repeat with elem in allElements
                        if elementCount >= {limit} then exit repeat
                        try
                            set elemName to name of elem
                            if elemName is missing value then set elemName to ""
                            set elemPos to position of elem
                            set elemSize to size of elem
                            set output to output & elemName & "|" & (item 1 of elemPos) & "|" & (item 2 of elemPos) & "|" & (item 1 of elemSize) & "|" & (item 2 of elemSize) & "\\n"
                            set elementCount to elementCount + 1
                        end try
                    end repeat
                    return output
                on error
                    return ""
                end try
            end tell
        end tell
        '''

        result = run_applescript(script)
        elements = []

        for line in result.strip().split('\n'):
            if not line or '|' not in line:
                continue
            parts = line.split('|')
            if len(parts) >= 5:
                try:
                    elements.append(UIElement(
                        name=parts[0] or "",
                        role=mapped_role,
                        position=(int(parts[1]), int(parts[2])),
                        size=(int(parts[3]), int(parts[4])),
                        app=app
                    ))
                except (ValueError, IndexError):
                    continue

        return elements

    def find_focused(self) -> Optional[UIElement]:
        """Get the currently focused element.

        Returns:
            The focused UIElement, or None if unable to determine
        """
        app = self.get_active_app()
        if not app:
            return None

        script = f'''
        tell application "System Events"
            tell process "{app}"
                try
                    set focusedElem to focused UI element of window 1
                    set elemName to name of focusedElem
                    if elemName is missing value then set elemName to ""
                    set elemPos to position of focusedElem
                    set elemSize to size of focusedElem
                    set elemRole to role of focusedElem
                    set elemValue to value of focusedElem
                    if elemValue is missing value then set elemValue to ""
                    return elemName & "|" & (item 1 of elemPos) & "|" & (item 2 of elemPos) & "|" & (item 1 of elemSize) & "|" & (item 2 of elemSize) & "|" & elemRole & "|" & elemValue
                on error
                    return ""
                end try
            end tell
        end tell
        '''

        result = run_applescript(script)
        if result and '|' in result:
            parts = result.split('|')
            if len(parts) >= 6:
                try:
                    return UIElement(
                        name=parts[0],
                        role=parts[5],
                        position=(int(parts[1]), int(parts[2])),
                        size=(int(parts[3]), int(parts[4])),
                        app=app,
                        is_focused=True,
                        value=parts[6] if len(parts) > 6 else None
                    )
                except (ValueError, IndexError):
                    pass

        return None

    def find_in_dock(self, name: str) -> Optional[UIElement]:
        """Find an element in the Dock.

        Args:
            name: Name of the dock item to find (case-insensitive, partial match)

        Returns:
            UIElement for the dock item, or None if not found
        """
        items = self.get_dock_items()
        name_lower = name.lower().strip()

        if not name_lower:
            return None

        # Try exact match first
        for item in items:
            if item.name.lower() == name_lower:
                return item

        # Try starts-with match
        for item in items:
            if item.name.lower().startswith(name_lower):
                return item

        # Try word-boundary match
        for item in items:
            item_words = item.name.lower().split()
            if name_lower in item_words:
                return item
            for word in item_words:
                if word.startswith(name_lower):
                    return item

        # Try contains match as last resort
        for item in items:
            if name_lower in item.name.lower():
                return item

        return None

    def get_dock_items(self) -> List[UIElement]:
        """Get all items in the Dock.

        Returns:
            List of UIElements representing dock items
        """
        cache_key = "dock_items"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        script = '''
        tell application "System Events"
            tell process "Dock"
                set dockItems to every UI element of list 1
                set output to ""
                repeat with item_ref in dockItems
                    try
                        set itemName to name of item_ref
                        set itemPos to position of item_ref
                        set itemSize to size of item_ref
                        set output to output & itemName & "|" & (item 1 of itemPos) & "|" & (item 2 of itemPos) & "|" & (item 1 of itemSize) & "|" & (item 2 of itemSize) & "\\n"
                    end try
                end repeat
                return output
            end tell
        end tell
        '''

        result = run_applescript(script)
        items = []

        for line in result.strip().split('\n'):
            if not line or '|' not in line:
                continue
            parts = line.split('|')
            if len(parts) >= 5:
                try:
                    name = parts[0]
                    # Skip separator items
                    if name == "missing value":
                        continue

                    items.append(UIElement(
                        name=name,
                        role="dock_item",
                        position=(int(parts[1]), int(parts[2])),
                        size=(int(parts[3]), int(parts[4])),
                        app="Dock"
                    ))
                except (ValueError, IndexError):
                    continue

        self._set_cached(cache_key, items)
        return items

    def find_in_menubar(self, name: str) -> Optional[UIElement]:
        """Find an element in the menu bar.

        Args:
            name: Name of the menu item to find

        Returns:
            UIElement for the menu item, or None if not found
        """
        items = self.get_menubar_items()
        name_lower = name.lower()

        for item in items:
            if item.name.lower() == name_lower:
                return item
            if name_lower in item.name.lower():
                return item

        return None

    def get_menubar_items(self) -> List[UIElement]:
        """Get all items in the menu bar of the frontmost application.

        Returns:
            List of UIElements representing menu bar items
        """
        app = self.get_active_app()

        script = '''
        tell application "System Events"
            set frontApp to first application process whose frontmost is true
            set menuBar to menu bar 1 of frontApp
            set menuItems to every menu bar item of menuBar
            set output to ""
            repeat with item_ref in menuItems
                try
                    set itemName to name of item_ref
                    set itemPos to position of item_ref
                    set itemSize to size of item_ref
                    set output to output & itemName & "|" & (item 1 of itemPos) & "|" & (item 2 of itemPos) & "|" & (item 1 of itemSize) & "|" & (item 2 of itemSize) & "\\n"
                end try
            end repeat
            return output
        end tell
        '''

        result = run_applescript(script)
        items = []

        for line in result.strip().split('\n'):
            if not line or '|' not in line:
                continue
            parts = line.split('|')
            if len(parts) >= 5:
                try:
                    items.append(UIElement(
                        name=parts[0],
                        role="menu_bar_item",
                        position=(int(parts[1]), int(parts[2])),
                        size=(int(parts[3]), int(parts[4])),
                        app=app or "System"
                    ))
                except (ValueError, IndexError):
                    continue

        return items

    def find_window(self, title: str, app: str = None) -> Optional[UIElement]:
        """Find a window by its title.

        Args:
            title: Title of the window to find (partial match)
            app: Application name (defaults to searching all apps)

        Returns:
            UIElement representing the window, or None if not found
        """
        if app:
            script = f'''
            tell application "System Events"
                tell process "{app}"
                    repeat with w in windows
                        try
                            set winTitle to name of w
                            if winTitle contains "{title}" then
                                set winPos to position of w
                                set winSize to size of w
                                return winTitle & "|" & (item 1 of winPos) & "|" & (item 2 of winPos) & "|" & (item 1 of winSize) & "|" & (item 2 of winSize)
                            end if
                        end try
                    end repeat
                end tell
            end tell
            return ""
            '''
        else:
            script = f'''
            tell application "System Events"
                repeat with proc in application processes
                    try
                        repeat with w in windows of proc
                            try
                                set winTitle to name of w
                                if winTitle contains "{title}" then
                                    set winPos to position of w
                                    set winSize to size of w
                                    set procName to name of proc
                                    return winTitle & "|" & (item 1 of winPos) & "|" & (item 2 of winPos) & "|" & (item 1 of winSize) & "|" & (item 2 of winSize) & "|" & procName
                                end if
                            end try
                        end repeat
                    end try
                end repeat
            end tell
            return ""
            '''

        result = run_applescript(script)
        if result and '|' in result:
            parts = result.split('|')
            if len(parts) >= 5:
                try:
                    return UIElement(
                        name=parts[0],
                        role="window",
                        position=(int(parts[1]), int(parts[2])),
                        size=(int(parts[3]), int(parts[4])),
                        app=parts[5] if len(parts) > 5 else (app or "unknown")
                    )
                except (ValueError, IndexError):
                    pass

        return None

    def get_windows(self, app: str = None) -> List[UIElement]:
        """Get all windows for an application.

        Args:
            app: Application name (defaults to frontmost app)

        Returns:
            List of UIElements representing windows
        """
        if not app:
            app = self.get_active_app()
        if not app:
            return []

        script = f'''
        tell application "System Events"
            tell process "{app}"
                set output to ""
                repeat with w in windows
                    try
                        set winTitle to name of w
                        if winTitle is missing value then set winTitle to "Untitled"
                        set winPos to position of w
                        set winSize to size of w
                        set output to output & winTitle & "|" & (item 1 of winPos) & "|" & (item 2 of winPos) & "|" & (item 1 of winSize) & "|" & (item 2 of winSize) & "\\n"
                    end try
                end repeat
                return output
            end tell
        end tell
        '''

        result = run_applescript(script)
        windows = []

        for line in result.strip().split('\n'):
            if not line or '|' not in line:
                continue
            parts = line.split('|')
            if len(parts) >= 5:
                try:
                    windows.append(UIElement(
                        name=parts[0],
                        role="window",
                        position=(int(parts[1]), int(parts[2])),
                        size=(int(parts[3]), int(parts[4])),
                        app=app
                    ))
                except (ValueError, IndexError):
                    continue

        return windows

    def get_all_elements(self, app: str = None,
                         window_index: int = 1,
                         limit: int = 100) -> List[UIElement]:
        """Get all accessible elements in an application window.

        Args:
            app: Application name (defaults to frontmost app)
            window_index: Which window to inspect (1-indexed)
            limit: Maximum number of elements to return

        Returns:
            List of all accessible UIElements
        """
        if not app:
            app = self.get_active_app()
        if not app:
            return []

        script = f'''
        tell application "System Events"
            tell process "{app}"
                try
                    set output to ""
                    set elementCount to 0
                    set allElements to entire contents of window {window_index}
                    repeat with elem in allElements
                        if elementCount >= {limit} then exit repeat
                        try
                            set elemName to name of elem
                            if elemName is missing value then set elemName to ""
                            set elemPos to position of elem
                            set elemSize to size of elem
                            set elemRole to role of elem
                            if elemRole is missing value then set elemRole to "unknown"
                            set output to output & elemName & "|" & (item 1 of elemPos) & "|" & (item 2 of elemPos) & "|" & (item 1 of elemSize) & "|" & (item 2 of elemSize) & "|" & elemRole & "\\n"
                            set elementCount to elementCount + 1
                        end try
                    end repeat
                    return output
                on error errMsg
                    return ""
                end try
            end tell
        end tell
        '''

        result = run_applescript(script, timeout=30)  # Longer timeout for this
        elements = []

        for line in result.strip().split('\n'):
            if not line or '|' not in line:
                continue
            parts = line.split('|')
            if len(parts) >= 6:
                try:
                    elements.append(UIElement(
                        name=parts[0],
                        role=parts[5],
                        position=(int(parts[1]), int(parts[2])),
                        size=(int(parts[3]), int(parts[4])),
                        app=app
                    ))
                except (ValueError, IndexError):
                    continue

        return elements

    def find_ui_element(self, description: str) -> Optional[UIElement]:
        """Find a UI element by natural language description.

        This tries multiple approaches:
        1. Check if it's a dock item
        2. Check if it's a menu item
        3. Check for special items (Apple menu, etc.)
        4. Search in the frontmost app

        Args:
            description: Natural language description like "Chrome icon",
                        "File menu", "Submit button"

        Returns:
            UIElement if found, None otherwise
        """
        desc_lower = description.lower()

        # Check for Apple menu first
        if 'apple' in desc_lower:
            return UIElement(
                name='Apple Menu',
                role='menu_bar_item',
                position=(10, 2),
                size=(20, 20),
                app='System'
            )

        # Check for menu items
        if 'menu' in desc_lower:
            menu_name = desc_lower.replace('menu', '').strip()
            item = self.find_in_menubar(menu_name)
            if item:
                return item

        # Clean up description to extract app/element name
        clean_desc = desc_lower

        # Remove filler phrases
        filler_patterns = [
            r'\bmove mouse to\b', r'\bmove the mouse to\b', r'\bmove to\b',
            r'\bclick on\b', r'\bclick the\b', r'\bclick\b',
            r'\bopen\b', r'\blaunch\b', r'\bstart\b',
            r'\bin the dock\b', r'\bin dock\b', r'\bfrom dock\b', r'\bfrom the dock\b',
            r'\bicon in\b', r'\bicons in\b',
            r'\bthe\b', r'\bin\b', r'\bto\b', r'\bon\b', r'\bfor\b', r'\bme\b',
            r'\bicon\b', r'\bicons\b',
            r'\bdock\b',
            r'\bapp\b', r'\bapplication\b',
            r'\bplease\b',
            r'\bbutton\b',
        ]
        for pattern in filler_patterns:
            clean_desc = re.sub(pattern, ' ', clean_desc)
        clean_desc = ' '.join(clean_desc.split()).strip()

        # Try dock search
        if clean_desc:
            item = self.find_in_dock(clean_desc)
            if item:
                return item

        # Try original description in dock
        item = self.find_in_dock(description)
        if item:
            return item

        # Try searching in the active app
        if clean_desc:
            item = self.find_by_name(clean_desc)
            if item:
                return item

        return None


# Singleton instance for convenience
_finder: Optional[ElementFinder] = None

def get_finder() -> ElementFinder:
    """Get the shared ElementFinder instance."""
    global _finder
    if _finder is None:
        _finder = ElementFinder()
    return _finder
