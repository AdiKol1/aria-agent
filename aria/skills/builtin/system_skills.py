"""
System Control Skills for Aria.

Provides desktop control capabilities:
- Opening applications
- Clicking, typing, scrolling
- Taking screenshots
- Getting active app info
"""

import re
from aria.skills.registry import skill
from aria.skills.base import SkillCategory, SkillContext, SkillResult


@skill(
    name="open_application",
    description="Open an application by name (e.g., 'open Chrome', 'launch Finder')",
    triggers=["open", "launch", "start", "run"],
    category=SkillCategory.SYSTEM,
)
def open_application(context: SkillContext) -> SkillResult:
    """Open an application."""
    from aria.control import open_app

    # Extract app name from input
    input_lower = context.user_input.lower()

    # Common app name patterns
    patterns = [
        r"open\s+(\w+)",
        r"launch\s+(\w+)",
        r"start\s+(\w+)",
        r"run\s+(\w+)",
    ]

    app_name = None
    for pattern in patterns:
        match = re.search(pattern, input_lower)
        if match:
            app_name = match.group(1)
            break

    if not app_name:
        return SkillResult.fail("Couldn't determine which app to open")

    # Map common names to actual app names
    app_map = {
        "chrome": "Google Chrome",
        "safari": "Safari",
        "finder": "Finder",
        "terminal": "Terminal",
        "vscode": "Visual Studio Code",
        "code": "Visual Studio Code",
        "slack": "Slack",
        "spotify": "Spotify",
        "notes": "Notes",
        "calendar": "Calendar",
        "mail": "Mail",
        "messages": "Messages",
        "photos": "Photos",
        "music": "Music",
        "preview": "Preview",
        "textedit": "TextEdit",
        "calculator": "Calculator",
    }

    actual_name = app_map.get(app_name.lower(), app_name.title())

    success = open_app(actual_name)
    if success:
        return SkillResult.ok(f"Opened {actual_name}")
    else:
        return SkillResult.fail(f"Failed to open {actual_name}")


@skill(
    name="click_at_position",
    description="Click at specific screen coordinates",
    triggers=["click at", "click on"],
    category=SkillCategory.SYSTEM,
    requires_screen=True,
)
def click_at_position(context: SkillContext) -> SkillResult:
    """Click at coordinates."""
    from aria.control import click

    # Extract coordinates from input
    input_text = context.user_input
    match = re.search(r"(\d+)\s*,?\s*(\d+)", input_text)

    if not match:
        return SkillResult.fail("Couldn't find coordinates. Use format: click at 100, 200")

    x, y = int(match.group(1)), int(match.group(2))

    success = click(x, y)
    if success:
        return SkillResult.ok(f"Clicked at ({x}, {y})")
    else:
        return SkillResult.fail(f"Failed to click at ({x}, {y})")


@skill(
    name="type_text",
    description="Type text at current cursor position",
    triggers=["type", "enter text", "write"],
    category=SkillCategory.SYSTEM,
)
def type_text_skill(context: SkillContext) -> SkillResult:
    """Type text."""
    from aria.control import type_text

    # Extract text to type
    input_text = context.user_input

    # Remove trigger words
    for trigger in ["type", "enter", "write"]:
        input_text = re.sub(rf"^\s*{trigger}\s+", "", input_text, flags=re.IGNORECASE)
        input_text = re.sub(rf"\s+{trigger}\s*$", "", input_text, flags=re.IGNORECASE)

    # Remove surrounding quotes
    text_to_type = input_text.strip().strip('"\'')

    if not text_to_type:
        return SkillResult.fail("No text specified to type")

    success = type_text(text_to_type)
    if success:
        return SkillResult.ok(f"Typed: {text_to_type[:50]}...")
    else:
        return SkillResult.fail("Failed to type text")


@skill(
    name="scroll_screen",
    description="Scroll up or down on the screen",
    triggers=["scroll up", "scroll down", "scroll"],
    category=SkillCategory.SYSTEM,
)
def scroll_screen(context: SkillContext) -> SkillResult:
    """Scroll the screen."""
    from aria.control import scroll

    input_lower = context.user_input.lower()

    # Determine direction and amount
    if "up" in input_lower:
        amount = 3
    elif "down" in input_lower:
        amount = -3
    else:
        amount = -3  # Default to scroll down

    # Check for amount modifier
    if "lot" in input_lower or "more" in input_lower or "much" in input_lower:
        amount *= 3

    success = scroll(amount)
    direction = "up" if amount > 0 else "down"
    if success:
        return SkillResult.ok(f"Scrolled {direction}")
    else:
        return SkillResult.fail("Failed to scroll")


@skill(
    name="press_keyboard_key",
    description="Press a keyboard key (enter, tab, escape, etc.)",
    triggers=["press", "hit", "key"],
    category=SkillCategory.SYSTEM,
)
def press_keyboard_key(context: SkillContext) -> SkillResult:
    """Press a keyboard key."""
    from aria.control import press_key

    input_lower = context.user_input.lower()

    # Map common key names
    key_map = {
        "enter": "enter",
        "return": "enter",
        "tab": "tab",
        "escape": "escape",
        "esc": "escape",
        "space": "space",
        "backspace": "backspace",
        "delete": "delete",
        "up": "up",
        "down": "down",
        "left": "left",
        "right": "right",
        "home": "home",
        "end": "end",
    }

    key = None
    for key_name, key_value in key_map.items():
        if key_name in input_lower:
            key = key_value
            break

    if not key:
        return SkillResult.fail("Couldn't determine which key to press")

    success = press_key(key)
    if success:
        return SkillResult.ok(f"Pressed {key}")
    else:
        return SkillResult.fail(f"Failed to press {key}")


@skill(
    name="keyboard_shortcut",
    description="Execute a keyboard shortcut (e.g., Cmd+C, Cmd+V)",
    triggers=["shortcut", "hotkey", "cmd", "command"],
    category=SkillCategory.SYSTEM,
)
def keyboard_shortcut(context: SkillContext) -> SkillResult:
    """Execute a keyboard shortcut."""
    from aria.control import hotkey

    input_lower = context.user_input.lower()

    # Common shortcuts
    shortcuts = {
        "copy": ["command", "c"],
        "paste": ["command", "v"],
        "cut": ["command", "x"],
        "undo": ["command", "z"],
        "redo": ["command", "shift", "z"],
        "save": ["command", "s"],
        "select all": ["command", "a"],
        "new": ["command", "n"],
        "close": ["command", "w"],
        "quit": ["command", "q"],
        "find": ["command", "f"],
        "print": ["command", "p"],
        "refresh": ["command", "r"],
        "tab": ["command", "tab"],
        "screenshot": ["command", "shift", "4"],
    }

    keys = None
    for shortcut_name, shortcut_keys in shortcuts.items():
        if shortcut_name in input_lower:
            keys = shortcut_keys
            break

    # Try to parse custom shortcut like "cmd+shift+s"
    if not keys:
        match = re.search(r"(cmd|command|ctrl|alt|shift)[\+\s]+([\w\+\s]+)", input_lower)
        if match:
            keys = []
            parts = match.group(0).replace("+", " ").split()
            for part in parts:
                part = part.strip()
                if part in ["cmd", "command"]:
                    keys.append("command")
                elif part in ["ctrl", "control"]:
                    keys.append("ctrl")
                elif part == "alt":
                    keys.append("alt")
                elif part == "shift":
                    keys.append("shift")
                else:
                    keys.append(part)

    if not keys:
        return SkillResult.fail("Couldn't determine keyboard shortcut")

    success = hotkey(keys)
    if success:
        return SkillResult.ok(f"Executed shortcut: {'+'.join(keys)}")
    else:
        return SkillResult.fail("Failed to execute shortcut")


@skill(
    name="take_screenshot",
    description="Take a screenshot of the current screen",
    triggers=["screenshot", "capture screen", "grab screen"],
    category=SkillCategory.SYSTEM,
)
def take_screenshot(context: SkillContext) -> SkillResult:
    """Take a screenshot."""
    from aria.vision import capture_screen

    screenshot = capture_screen()
    if screenshot:
        return SkillResult.ok(
            "Captured screenshot",
            data={"screenshot": screenshot}
        )
    else:
        return SkillResult.fail("Failed to capture screenshot")


@skill(
    name="get_active_application",
    description="Get the name of the currently active/focused application",
    triggers=["active app", "current app", "focused app", "what app"],
    category=SkillCategory.SYSTEM,
)
def get_active_application(context: SkillContext) -> SkillResult:
    """Get the active application."""
    from aria.control import get_active_app

    app_name = get_active_app()
    if app_name:
        return SkillResult.ok(
            f"The active application is {app_name}",
            data={"app_name": app_name}
        )
    else:
        return SkillResult.fail("Couldn't determine active application")


@skill(
    name="open_url",
    description="Open a URL in the default browser",
    triggers=["go to", "open url", "visit", "browse"],
    category=SkillCategory.BROWSER,
)
def open_url_skill(context: SkillContext) -> SkillResult:
    """Open a URL in the browser."""
    from aria.control import open_url

    input_text = context.user_input

    # Extract URL
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    match = re.search(url_pattern, input_text)

    if match:
        url = match.group(0)
    else:
        # Try to construct URL from domain-like text
        domain_pattern = r'(?:go to|visit|open|browse)\s+([a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,})'
        match = re.search(domain_pattern, input_text, re.IGNORECASE)
        if match:
            url = f"https://{match.group(1)}"
        else:
            return SkillResult.fail("Couldn't find a URL to open")

    success = open_url(url)
    if success:
        return SkillResult.ok(f"Opened {url}")
    else:
        return SkillResult.fail(f"Failed to open {url}")
