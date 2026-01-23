"""Command Executor - Bridges intents to accessibility actions.

This module takes parsed Intent objects and executes them using the
Accessibility API, returning IntentResult objects.

The CommandExecutor is the bridge between the high-level intent parsing
system and the low-level accessibility actions. It handles:
- Routing intents to appropriate handlers
- Error handling and recovery
- Converting accessibility results to IntentResult objects

Example:
    from aria.core.executor import CommandExecutor
    from aria.core.intent_parser import parse

    executor = CommandExecutor()
    intent = parse("click on Chrome")
    result = executor.execute(intent)
    print(result.response)  # "Clicked on Google Chrome"
"""
import logging
from typing import Optional, Dict, Any, Callable

from ..intents.base import Intent, IntentType, IntentResult
from ..accessibility import (
    ElementFinder, UIActions, get_finder, get_actions,
    ChromeHandler, SafariHandler, FinderHandler, SystemHandler,
    get_browser_handler,
)

# Set up logging
logger = logging.getLogger(__name__)


class CommandExecutor:
    """Executes intents using the Accessibility API.

    This class routes parsed intents to the appropriate accessibility handlers
    and returns results. It handles all supported intent types including:
    - CLICK: Mouse click actions
    - OPEN: Opening apps, files, URLs
    - TYPE: Typing text
    - SCROLL: Scrolling actions
    - CLOSE: Closing apps, windows, tabs
    - TAB: Tab management
    - KEYBOARD: Keyboard shortcuts
    - NAVIGATE: URL navigation

    Attributes:
        _finder: ElementFinder instance for locating UI elements
        _actions: UIActions instance for performing actions
        _chrome: ChromeHandler for Chrome-specific operations
        _safari: SafariHandler for Safari-specific operations
        _finder_app: FinderHandler for Finder operations
        _system: SystemHandler for system-wide operations

    Example:
        executor = CommandExecutor()

        # Execute a click intent
        intent = Intent(
            action=IntentType.CLICK,
            target="Chrome",
            params={"location": "dock"},
            confidence=0.95
        )
        result = executor.execute(intent)

        if result.success:
            print(result.response)  # "Clicked on Google Chrome"
    """

    def __init__(self):
        """Initialize the command executor with accessibility handlers."""
        self._finder = get_finder()
        self._actions = get_actions()
        self._chrome = ChromeHandler()
        self._safari = SafariHandler()
        self._finder_app = FinderHandler()
        self._system = SystemHandler()

        # Map intent types to handler methods
        self._handlers: Dict[IntentType, Callable[[Intent], IntentResult]] = {
            IntentType.CLICK: self._execute_click,
            IntentType.OPEN: self._execute_open,
            IntentType.TYPE: self._execute_type,
            IntentType.SCROLL: self._execute_scroll,
            IntentType.TAB: self._execute_tab,
            IntentType.KEYBOARD: self._execute_keyboard,
            IntentType.NAVIGATE: self._execute_navigate,
            IntentType.CLOSE: self._execute_close,
        }

    def execute(self, intent: Intent) -> IntentResult:
        """Execute an intent and return the result.

        Routes the intent to the appropriate handler based on its action type.
        Handles errors gracefully and returns an IntentResult with success/failure
        status and descriptive response.

        Args:
            intent: The parsed intent to execute.

        Returns:
            IntentResult with success/failure status, response message,
            optional data, and error information if applicable.

        Raises:
            No exceptions are raised; all errors are captured in the result.
        """
        logger.debug(f"Executing intent: {intent.action.value} with target: {intent.target}")

        handler = self._handlers.get(intent.action)
        if handler:
            try:
                result = handler(intent)
                logger.debug(f"Intent execution result: success={result.success}")
                return result
            except Exception as e:
                logger.error(f"Error executing {intent.action.value}: {e}", exc_info=True)
                return IntentResult.error_result(
                    error=str(e),
                    response=f"Error executing {intent.action.value}: {str(e)}"
                )

        # Handle unsupported intent types
        logger.warning(f"Unknown intent type: {intent.action}")
        return IntentResult.error_result(
            error="UNKNOWN_INTENT",
            response=f"Unknown intent type: {intent.action.value}"
        )

    def _execute_click(self, intent: Intent) -> IntentResult:
        """Execute a click intent.

        Finds the target UI element and performs a click action. Supports
        single click, double-click, and right-click modifiers.

        Args:
            intent: Click intent with target and optional modifiers.

        Returns:
            IntentResult indicating success or failure of the click.
        """
        target = intent.target
        if not target:
            return IntentResult.error_result(
                error="NO_TARGET",
                response="Click target not specified"
            )

        # Try to find the element using natural language description
        element = self._finder.find_ui_element(target)

        if element:
            # Determine click type based on modifiers
            if intent.params.get("double_click"):
                success = self._actions.double_click(element)
                action_name = "Double-clicked"
            elif intent.params.get("right_click"):
                success = self._actions.right_click(element)
                action_name = "Right-clicked"
            else:
                success = self._actions.click(element)
                action_name = "Clicked"

            if success:
                return IntentResult.success_result(
                    response=f"{action_name} on {element.name}",
                    data={"element": element.to_dict()}
                )
            else:
                return IntentResult.error_result(
                    error="CLICK_FAILED",
                    response=f"Failed to click on {element.name}"
                )

        return IntentResult.error_result(
            error="ELEMENT_NOT_FOUND",
            response=f"Could not find element: {target}"
        )

    def _execute_open(self, intent: Intent) -> IntentResult:
        """Execute an open intent (open app, file, or URL).

        Determines whether the target is a URL, app, or file and uses
        the appropriate method to open it.

        Args:
            intent: Open intent with target and optional type hints.

        Returns:
            IntentResult indicating success or failure.
        """
        target = intent.target
        if not target:
            return IntentResult.error_result(
                error="NO_TARGET",
                response="Open target not specified"
            )

        # Check if it's a URL
        is_url = (
            intent.params.get("is_url") or
            target.startswith(("http://", "https://", "www.")) or
            self._looks_like_url(target)
        )

        if is_url:
            # Ensure URL has protocol
            url = target if target.startswith(("http://", "https://")) else f"https://{target}"
            success = self._system.open_url(url)
            if success:
                return IntentResult.success_result(
                    response=f"Opened URL: {url}",
                    data={"url": url, "type": "url"}
                )
            else:
                return IntentResult.error_result(
                    error="URL_OPEN_FAILED",
                    response=f"Failed to open URL: {url}"
                )

        # Otherwise, treat as app
        app_name = intent.params.get("app_name", target)

        # Try common app name mappings
        app_name = self._resolve_app_name(app_name)

        success = self._system.open_app(app_name)
        if success:
            return IntentResult.success_result(
                response=f"Opened {app_name}",
                data={"app": app_name, "type": "app"}
            )
        else:
            return IntentResult.error_result(
                error="APP_OPEN_FAILED",
                response=f"Failed to open {app_name}"
            )

    def _execute_type(self, intent: Intent) -> IntentResult:
        """Execute a type intent.

        Types the specified text. If the text ends with "and enter" or
        similar, also presses the Enter key afterward.

        Args:
            intent: Type intent with text to type in params or target.

        Returns:
            IntentResult indicating success or failure.
        """
        # Get text from params or target
        text = intent.params.get("text") or intent.target

        if not text:
            return IntentResult.error_result(
                error="NO_TEXT",
                response="No text specified to type"
            )

        # Check for "and enter" suffix
        should_press_enter = False
        text_lower = text.lower()
        enter_suffixes = [" and enter", " and press enter", " then enter", " and hit enter"]

        for suffix in enter_suffixes:
            if text_lower.endswith(suffix):
                text = text[:-len(suffix)]
                should_press_enter = True
                break

        # Type the text
        success = self._actions.type_text(text)

        if not success:
            return IntentResult.error_result(
                error="TYPE_FAILED",
                response=f"Failed to type text"
            )

        # Press enter if needed
        if should_press_enter:
            self._actions.press_key("enter")

        # Truncate text for display if too long
        display_text = text[:50] + "..." if len(text) > 50 else text
        response = f"Typed: {display_text}"
        if should_press_enter:
            response += " and pressed Enter"

        return IntentResult.success_result(
            response=response,
            data={"text": text, "pressed_enter": should_press_enter}
        )

    def _execute_scroll(self, intent: Intent) -> IntentResult:
        """Execute a scroll intent.

        Scrolls up or down based on direction and amount specified in params.

        Args:
            intent: Scroll intent with direction and optional amount.

        Returns:
            IntentResult indicating success or failure.
        """
        direction = intent.params.get("direction", "down")
        amount = intent.params.get("amount", 3)  # Default scroll amount

        # Handle special directions
        if direction == "top":
            # Scroll to top - use a large amount
            amount = 100
            direction = "up"
        elif direction == "bottom":
            # Scroll to bottom - use a large amount
            amount = 100
            direction = "down"

        # Convert direction to scroll amount (positive = up, negative = down)
        if direction in ["down", "right"]:
            scroll_amount = -abs(amount)
        else:
            scroll_amount = abs(amount)

        success = self._actions.scroll(scroll_amount)

        if success:
            return IntentResult.success_result(
                response=f"Scrolled {direction}",
                data={"direction": direction, "amount": amount}
            )
        else:
            return IntentResult.error_result(
                error="SCROLL_FAILED",
                response=f"Failed to scroll {direction}"
            )

    def _execute_tab(self, intent: Intent) -> IntentResult:
        """Execute a tab management intent.

        Handles new tab, close tab, switch tab, next/previous tab operations.
        Works with both Chrome and Safari based on the active app.

        Args:
            intent: Tab intent with action type and optional parameters.

        Returns:
            IntentResult indicating success or failure.
        """
        action = intent.params.get("action", "new")
        tab_number = intent.params.get("tab_number")

        # Get the appropriate browser handler
        browser = get_browser_handler()

        if action == "new":
            result = browser.new_tab()
            if result.get("success"):
                return IntentResult.success_result(
                    response="Opened new tab",
                    data={"action": "new_tab"}
                )
            else:
                return IntentResult.error_result(
                    error="TAB_NEW_FAILED",
                    response="Failed to open new tab"
                )

        elif action == "close":
            result = browser.close_tab()
            if result.get("success"):
                closed_tab = result.get("closed_tab", "tab")
                return IntentResult.success_result(
                    response=f"Closed tab: {closed_tab}",
                    data={"action": "close_tab", "closed_tab": closed_tab}
                )
            else:
                return IntentResult.error_result(
                    error="TAB_CLOSE_FAILED",
                    response=result.get("message", "Failed to close tab")
                )

        elif action == "next":
            # Use keyboard shortcut for next tab
            success = self._actions.hotkey("command", "shift", "]")
            if success:
                return IntentResult.success_result(
                    response="Switched to next tab",
                    data={"action": "next_tab"}
                )
            else:
                return IntentResult.error_result(
                    error="TAB_SWITCH_FAILED",
                    response="Failed to switch to next tab"
                )

        elif action == "previous":
            # Use keyboard shortcut for previous tab
            success = self._actions.hotkey("command", "shift", "[")
            if success:
                return IntentResult.success_result(
                    response="Switched to previous tab",
                    data={"action": "previous_tab"}
                )
            else:
                return IntentResult.error_result(
                    error="TAB_SWITCH_FAILED",
                    response="Failed to switch to previous tab"
                )

        elif action == "switch" and tab_number is not None:
            # Switch to specific tab by number
            if 1 <= tab_number <= 9:
                success = self._actions.hotkey("command", str(tab_number))
                if success:
                    return IntentResult.success_result(
                        response=f"Switched to tab {tab_number}",
                        data={"action": "switch_tab", "tab_number": tab_number}
                    )
            # For tabs > 9, use the browser API
            result = browser.switch_tab(index=tab_number)
            if result.get("success"):
                return IntentResult.success_result(
                    response=f"Switched to tab {tab_number}",
                    data={"action": "switch_tab", "tab_number": tab_number}
                )
            else:
                return IntentResult.error_result(
                    error="TAB_SWITCH_FAILED",
                    response=result.get("message", f"Failed to switch to tab {tab_number}")
                )

        elif action == "reopen":
            # Reopen last closed tab
            success = self._actions.hotkey("command", "shift", "t")
            if success:
                return IntentResult.success_result(
                    response="Reopened last closed tab",
                    data={"action": "reopen_tab"}
                )
            else:
                return IntentResult.error_result(
                    error="TAB_REOPEN_FAILED",
                    response="Failed to reopen last closed tab"
                )

        return IntentResult.error_result(
            error="UNKNOWN_TAB_ACTION",
            response=f"Unknown tab action: {action}"
        )

    def _execute_keyboard(self, intent: Intent) -> IntentResult:
        """Execute a keyboard shortcut intent.

        Presses the specified key or key combination.

        Args:
            intent: Keyboard intent with keys to press.

        Returns:
            IntentResult indicating success or failure.
        """
        keys = intent.params.get("keys", [])

        if not keys:
            return IntentResult.error_result(
                error="NO_KEYS",
                response="No keys specified for keyboard shortcut"
            )

        # Normalize key names
        normalized_keys = []
        for key in keys:
            key_lower = key.lower()
            # Map common key names
            if key_lower in ["cmd", "command"]:
                normalized_keys.append("command")
            elif key_lower in ["ctrl", "control"]:
                normalized_keys.append("ctrl")
            elif key_lower in ["alt", "option"]:
                normalized_keys.append("option")
            elif key_lower == "shift":
                normalized_keys.append("shift")
            elif key_lower in ["enter", "return"]:
                normalized_keys.append("return")
            elif key_lower in ["esc", "escape"]:
                normalized_keys.append("escape")
            else:
                normalized_keys.append(key_lower)

        # Single key press vs hotkey
        if len(normalized_keys) == 1 and normalized_keys[0] not in ["command", "ctrl", "option", "shift"]:
            success = self._actions.press_key(normalized_keys[0])
        else:
            success = self._actions.hotkey(*normalized_keys)

        if success:
            key_display = "+".join(keys)
            return IntentResult.success_result(
                response=f"Pressed {key_display}",
                data={"keys": keys}
            )
        else:
            return IntentResult.error_result(
                error="KEYBOARD_FAILED",
                response=f"Failed to press keyboard shortcut"
            )

    def _execute_navigate(self, intent: Intent) -> IntentResult:
        """Execute a URL navigation intent.

        Navigates the current browser tab to the specified URL.

        Args:
            intent: Navigate intent with URL target.

        Returns:
            IntentResult indicating success or failure.
        """
        target = intent.target
        if not target:
            return IntentResult.error_result(
                error="NO_URL",
                response="No URL specified for navigation"
            )

        # Ensure URL has protocol
        url = target if target.startswith(("http://", "https://")) else f"https://{target}"

        # Get active browser and navigate
        browser = get_browser_handler()
        success = browser.navigate(url)

        if success:
            return IntentResult.success_result(
                response=f"Navigated to {url}",
                data={"url": url}
            )
        else:
            return IntentResult.error_result(
                error="NAVIGATE_FAILED",
                response=f"Failed to navigate to {url}"
            )

    def _execute_close(self, intent: Intent) -> IntentResult:
        """Execute a close intent.

        Closes the specified app, window, or tab.

        Args:
            intent: Close intent with target specification.

        Returns:
            IntentResult indicating success or failure.
        """
        target = intent.target

        # If no target, close current window
        if not target:
            success = self._actions.close_window()
            if success:
                return IntentResult.success_result(
                    response="Closed current window",
                    data={"type": "window"}
                )
            else:
                return IntentResult.error_result(
                    error="CLOSE_FAILED",
                    response="Failed to close current window"
                )

        target_lower = target.lower()

        # Check if it's a tab close
        if "tab" in target_lower:
            return self._execute_tab(Intent(
                action=IntentType.TAB,
                target=target,
                params={"action": "close"},
                confidence=intent.confidence,
                raw_text=intent.raw_text
            ))

        # Check if it's a window close
        if "window" in target_lower:
            success = self._actions.close_window()
            if success:
                return IntentResult.success_result(
                    response="Closed window",
                    data={"type": "window"}
                )
            else:
                return IntentResult.error_result(
                    error="CLOSE_FAILED",
                    response="Failed to close window"
                )

        # Try to quit the app
        app_name = self._resolve_app_name(target)
        success = self._system.quit_app(app_name)

        if success:
            return IntentResult.success_result(
                response=f"Closed {app_name}",
                data={"app": app_name, "type": "app"}
            )
        else:
            return IntentResult.error_result(
                error="CLOSE_FAILED",
                response=f"Failed to close {app_name}"
            )

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _looks_like_url(self, text: str) -> bool:
        """Check if text looks like a URL.

        Args:
            text: Text to check.

        Returns:
            True if text appears to be a URL.
        """
        # Check for common URL patterns
        url_indicators = [
            ".com", ".org", ".net", ".io", ".dev", ".app",
            ".edu", ".gov", ".co", ".ai", ".me",
            "localhost", "127.0.0.1"
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in url_indicators)

    def _resolve_app_name(self, name: str) -> str:
        """Resolve common app name aliases to full app names.

        Args:
            name: App name or alias.

        Returns:
            Full app name for AppleScript.
        """
        # Common app name mappings
        app_aliases = {
            "chrome": "Google Chrome",
            "googlechrome": "Google Chrome",
            "google chrome": "Google Chrome",
            "safari": "Safari",
            "finder": "Finder",
            "terminal": "Terminal",
            "iterm": "iTerm",
            "iterm2": "iTerm",
            "code": "Visual Studio Code",
            "vscode": "Visual Studio Code",
            "vs code": "Visual Studio Code",
            "visual studio code": "Visual Studio Code",
            "slack": "Slack",
            "spotify": "Spotify",
            "discord": "Discord",
            "firefox": "Firefox",
            "notes": "Notes",
            "mail": "Mail",
            "messages": "Messages",
            "imessage": "Messages",
            "calendar": "Calendar",
            "music": "Music",
            "itunes": "Music",
            "photos": "Photos",
            "preview": "Preview",
            "textedit": "TextEdit",
            "text edit": "TextEdit",
            "system preferences": "System Preferences",
            "settings": "System Preferences",
            "system settings": "System Settings",
            "activity monitor": "Activity Monitor",
            "zoom": "zoom.us",
            "teams": "Microsoft Teams",
            "microsoft teams": "Microsoft Teams",
            "word": "Microsoft Word",
            "excel": "Microsoft Excel",
            "powerpoint": "Microsoft PowerPoint",
            "notion": "Notion",
            "figma": "Figma",
            "sketch": "Sketch",
            "xcode": "Xcode",
            "sublime": "Sublime Text",
            "sublime text": "Sublime Text",
            "atom": "Atom",
        }

        name_lower = name.lower().strip()
        return app_aliases.get(name_lower, name)


# Module-level convenience function
def execute_intent(intent: Intent) -> IntentResult:
    """Execute an intent using the default executor.

    Convenience function for one-off intent execution.

    Args:
        intent: The intent to execute.

    Returns:
        IntentResult with execution outcome.
    """
    executor = CommandExecutor()
    return executor.execute(intent)
