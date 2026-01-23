"""
Accessibility API module for Aria.

This module provides a comprehensive interface to macOS Accessibility APIs,
enabling UI element discovery, interaction, and app-specific functionality.

Main Components:
- UIElement: Dataclass representing a UI element with position, size, and properties
- ElementFinder: Discover UI elements in apps, dock, menu bar, etc.
- UIActions: Perform actions like click, type, scroll, hotkeys
- App Handlers: Specialized handlers for Chrome, Safari, Finder, Terminal

Example Usage:
    from aria.accessibility import ElementFinder, UIActions, SystemHandler

    # Find and click a button
    finder = ElementFinder()
    button = finder.find_by_name("Submit", app="Safari")
    if button:
        actions = UIActions()
        actions.click(button)

    # Work with dock
    dock_item = finder.find_in_dock("Chrome")
    if dock_item:
        actions.click(dock_item)

    # Browser tab management
    from aria.accessibility import ChromeHandler
    chrome = ChromeHandler()
    tabs = chrome.get_tabs()
    chrome.switch_tab(title="GitHub")
    chrome.close_tab(title="YouTube")

    # System operations
    system = SystemHandler()
    system.show_notification("Hello!", "This is a notification")
    system.open_url("https://example.com")

Backward Compatibility:
    All functions from the original accessibility.py are available:
    - get_dock_items(), find_dock_item(), click_dock_item(), list_dock_items()
    - get_menu_bar_items(), find_menu_item()
    - find_ui_element()
    - get_browser_tabs(), close_browser_tab(), switch_browser_tab(), etc.
"""

from typing import Optional, Dict, List, Tuple
import re

# Core element classes
from .elements import (
    UIElement,
    ElementFinder,
    run_applescript,
    get_finder,
)

# Action classes
from .actions import (
    UIActions,
    get_actions,
)

# App-specific handlers
from .apps import (
    ChromeHandler,
    SafariHandler,
    FinderHandler,
    SystemHandler,
    TerminalHandler,
)


# ============================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# These functions maintain compatibility with the original
# aria/accessibility.py API
# ============================================================

def get_dock_items() -> List[Dict]:
    """Get all dock items with their positions and sizes.

    Returns:
        List of dicts with 'name', 'x', 'y', 'width', 'height', 'center_x', 'center_y'
    """
    finder = get_finder()
    elements = finder.get_dock_items()

    items = []
    for elem in elements:
        items.append({
            'name': elem.name,
            'x': elem.position[0],
            'y': elem.position[1],
            'width': elem.size[0],
            'height': elem.size[1],
            'center_x': elem.center_x,
            'center_y': elem.center_y,
        })

    return items


def find_dock_item(name: str) -> Optional[Dict]:
    """Find a dock item by name (case-insensitive partial match).

    Args:
        name: Name to search for (e.g., "chrome", "finder", "trash")

    Returns:
        Dict with item info including center_x, center_y, or None if not found
    """
    finder = get_finder()
    elem = finder.find_in_dock(name)

    if elem:
        return {
            'name': elem.name,
            'x': elem.position[0],
            'y': elem.position[1],
            'width': elem.size[0],
            'height': elem.size[1],
            'center_x': elem.center_x,
            'center_y': elem.center_y,
        }

    return None


def get_menu_bar_items() -> List[Dict]:
    """Get menu bar items for the frontmost application.

    Returns:
        List of dicts with 'name', 'x', 'y', 'width', 'height', 'center_x', 'center_y'
    """
    finder = get_finder()
    elements = finder.get_menubar_items()

    items = []
    for elem in elements:
        items.append({
            'name': elem.name,
            'x': elem.position[0],
            'y': elem.position[1],
            'width': elem.size[0],
            'height': elem.size[1],
            'center_x': elem.center_x,
            'center_y': elem.center_y,
        })

    return items


def find_menu_item(name: str) -> Optional[Dict]:
    """Find a menu bar item by name.

    Args:
        name: Menu name (e.g., "File", "Edit", "View")

    Returns:
        Dict with item info or None
    """
    finder = get_finder()
    elem = finder.find_in_menubar(name)

    if elem:
        return {
            'name': elem.name,
            'x': elem.position[0],
            'y': elem.position[1],
            'width': elem.size[0],
            'height': elem.size[1],
            'center_x': elem.center_x,
            'center_y': elem.center_y,
        }

    return None


def get_apple_menu_position() -> Tuple[int, int]:
    """Get the Apple menu position (always top-left).

    Returns:
        (x, y) tuple for center of Apple menu
    """
    # Apple menu is always at approximately (20, 12) on macOS
    return (20, 12)


def find_ui_element(description: str) -> Optional[Dict]:
    """Find a UI element by description.

    This tries multiple approaches:
    1. Check if it's a dock item
    2. Check if it's a menu item
    3. Check for special items (Apple menu, etc.)

    Args:
        description: Description like "Chrome icon", "Finder in dock", "File menu"

    Returns:
        Dict with 'x', 'y', 'center_x', 'center_y', 'name' or None
    """
    finder = get_finder()
    elem = finder.find_ui_element(description)

    if elem:
        return {
            'name': elem.name,
            'x': elem.position[0],
            'y': elem.position[1],
            'width': elem.size[0],
            'height': elem.size[1],
            'center_x': elem.center_x,
            'center_y': elem.center_y,
        }

    return None


def click_dock_item(name: str) -> bool:
    """Click on a dock item by name.

    Args:
        name: Name of the dock item

    Returns:
        True if clicked successfully
    """
    finder = get_finder()
    actions = get_actions()

    item = finder.find_in_dock(name)
    if not item:
        print(f"[Accessibility] Dock item not found: {name}")
        return False

    result = actions.click(item)
    if result:
        print(f"[Accessibility] Clicked {item.name} at ({item.center_x}, {item.center_y})")
    return result


def list_dock_items() -> str:
    """Get a formatted list of all dock items for display."""
    items = get_dock_items()
    if not items:
        return "No dock items found"

    lines = ["Dock items:"]
    for item in items:
        lines.append(f"  - {item['name']} at ({item['center_x']}, {item['center_y']})")
    return '\n'.join(lines)


# ============================================================
# BROWSER TAB CONTROL
# ============================================================

def get_browser_tabs(browser: str = "Google Chrome") -> List[Dict]:
    """Get all tabs from the specified browser.

    Args:
        browser: "Google Chrome" or "Safari"

    Returns:
        List of dicts with 'title', 'url', 'index', 'window_index'
    """
    if browser == "Google Chrome":
        handler = ChromeHandler()
    elif browser == "Safari":
        handler = SafariHandler()
    else:
        return []

    return handler.get_tabs()


def close_browser_tab(tab_identifier: str = None, browser: str = "Google Chrome") -> Dict:
    """Close a browser tab by title/URL match or close current tab.

    Args:
        tab_identifier: Part of tab title or URL to match. If None, closes current tab.
        browser: "Google Chrome" or "Safari"

    Returns:
        Dict with 'success', 'message', and optionally 'closed_tab'
    """
    if browser == "Google Chrome":
        handler = ChromeHandler()
    elif browser == "Safari":
        handler = SafariHandler()
    else:
        return {'success': False, 'message': f'Unknown browser: {browser}'}

    return handler.close_tab(title=tab_identifier)


def switch_browser_tab(tab_identifier: str, browser: str = "Google Chrome") -> Dict:
    """Switch to a browser tab by title/URL match.

    Args:
        tab_identifier: Part of tab title or URL to match
        browser: "Google Chrome" or "Safari"

    Returns:
        Dict with 'success', 'message'
    """
    if browser == "Google Chrome":
        handler = ChromeHandler()
    elif browser == "Safari":
        handler = SafariHandler()
    else:
        return {'success': False, 'message': f'Unknown browser: {browser}'}

    return handler.switch_tab(title=tab_identifier)


def list_browser_tabs(browser: str = "Google Chrome") -> str:
    """Get a formatted list of all browser tabs.

    Args:
        browser: "Google Chrome" or "Safari"

    Returns:
        Formatted string listing all tabs
    """
    tabs = get_browser_tabs(browser)
    if not tabs:
        return f"No tabs found in {browser}"

    lines = [f"{browser} tabs:"]
    for i, tab in enumerate(tabs, 1):
        title = tab['title'][:50] + '...' if len(tab['title']) > 50 else tab['title']
        lines.append(f"  {i}. {title}")

    return '\n'.join(lines)


def close_tab_by_position(position: str, browser: str = "Google Chrome") -> Dict:
    """Close a tab by its position (e.g., 'second from right', 'third tab').

    Args:
        position: Description like 'second from right', 'third tab', 'last tab'
        browser: "Google Chrome" or "Safari"

    Returns:
        Dict with 'success', 'message'
    """
    tabs = get_browser_tabs(browser)
    if not tabs:
        return {'success': False, 'message': 'No tabs found'}

    # Get tabs from front window only
    front_window_tabs = [t for t in tabs if t['window_index'] == 1]
    if not front_window_tabs:
        return {'success': False, 'message': 'No tabs in front window'}

    position_lower = position.lower()
    target_index = None

    # Parse position
    ordinals = {
        'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
        '1st': 1, '2nd': 2, '3rd': 3, '4th': 4, '5th': 5,
        'last': -1
    }

    from_right = 'right' in position_lower or 'end' in position_lower

    for word, idx in ordinals.items():
        if word in position_lower:
            if idx == -1:
                target_index = len(front_window_tabs)
            elif from_right:
                target_index = len(front_window_tabs) - idx + 1
            else:
                target_index = idx
            break

    if target_index is None:
        # Try to parse a number
        numbers = re.findall(r'\d+', position)
        if numbers:
            num = int(numbers[0])
            if from_right:
                target_index = len(front_window_tabs) - num + 1
            else:
                target_index = num

    if target_index is None or target_index < 1 or target_index > len(front_window_tabs):
        return {'success': False, 'message': f'Could not parse position: {position}. Found {len(front_window_tabs)} tabs.'}

    # Close the tab
    tab_to_close = front_window_tabs[target_index - 1]

    if browser == "Google Chrome":
        handler = ChromeHandler()
    else:
        handler = SafariHandler()

    return handler.close_tab(index=tab_to_close['index'], window_index=tab_to_close.get('window_index', 1))


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_active_app() -> str:
    """Get the name of the currently active application.

    Returns:
        Application name
    """
    finder = get_finder()
    return finder.get_active_app()


def click_at(x: int, y: int) -> bool:
    """Click at specific coordinates.

    Args:
        x: X coordinate
        y: Y coordinate

    Returns:
        True if click was successful
    """
    actions = get_actions()
    return actions.click((x, y))


def type_text(text: str) -> bool:
    """Type text at the current cursor position.

    Args:
        text: Text to type

    Returns:
        True if typing was successful
    """
    actions = get_actions()
    return actions.type_text(text)


def press_key(key: str) -> bool:
    """Press a key.

    Args:
        key: Key to press (enter, tab, escape, etc.)

    Returns:
        True if key press was successful
    """
    actions = get_actions()
    return actions.press_key(key)


def hotkey(*keys: str) -> bool:
    """Press a keyboard shortcut.

    Args:
        *keys: Keys to press together (e.g., 'command', 'c')

    Returns:
        True if hotkey was successful
    """
    actions = get_actions()
    return actions.hotkey(*keys)


def scroll(amount: int) -> bool:
    """Scroll up or down.

    Args:
        amount: Positive for up, negative for down

    Returns:
        True if scroll was successful
    """
    actions = get_actions()
    return actions.scroll(amount)


# ============================================================
# BROWSER HANDLER HELPER
# ============================================================

def get_browser_handler(browser: str = None):
    """Get the appropriate browser handler.

    Args:
        browser: Browser name ("chrome", "safari") or None for auto-detect

    Returns:
        ChromeHandler or SafariHandler instance
    """
    if browser is None:
        # Auto-detect based on running apps
        finder = get_finder()
        active_app = finder.get_active_app()
        if "Chrome" in active_app:
            return ChromeHandler()
        elif "Safari" in active_app:
            return SafariHandler()
        # Default to Chrome if can't detect
        return ChromeHandler()

    browser_lower = browser.lower()
    if "chrome" in browser_lower:
        return ChromeHandler()
    elif "safari" in browser_lower:
        return SafariHandler()
    else:
        # Default to Chrome
        return ChromeHandler()


__all__ = [
    # Core elements
    'UIElement',
    'ElementFinder',
    'run_applescript',
    'get_finder',

    # Actions
    'UIActions',
    'get_actions',

    # App handlers
    'ChromeHandler',
    'SafariHandler',
    'FinderHandler',
    'SystemHandler',
    'TerminalHandler',

    # Convenience functions
    'get_browser_handler',

    # Backward compatible functions
    'get_dock_items',
    'find_dock_item',
    'get_menu_bar_items',
    'find_menu_item',
    'get_apple_menu_position',
    'find_ui_element',
    'click_dock_item',
    'list_dock_items',
    'get_browser_tabs',
    'close_browser_tab',
    'switch_browser_tab',
    'list_browser_tabs',
    'close_tab_by_position',
    'get_active_app',
    'click_at',
    'type_text',
    'press_key',
    'hotkey',
    'scroll',
]
