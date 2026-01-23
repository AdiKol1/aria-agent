"""
App-specific accessibility handlers.

This module provides specialized handlers for common applications,
offering app-aware methods for browser tabs, file management, and system control.
"""
from typing import Optional, List, Dict, Any
import os
import re
import time

from .elements import UIElement, run_applescript, get_finder


class ChromeHandler:
    """Chrome-specific accessibility handler.

    Provides methods for interacting with Google Chrome tabs, windows,
    and navigation.

    Example:
        chrome = ChromeHandler()
        tabs = chrome.get_tabs()
        chrome.switch_tab(title="GitHub")
    """

    def __init__(self):
        """Initialize the Chrome handler."""
        self.app_name = "Google Chrome"

    def is_running(self) -> bool:
        """Check if Chrome is running."""
        script = '''
        tell application "System Events"
            return (name of processes) contains "Google Chrome"
        end tell
        '''
        result = run_applescript(script)
        return result.lower() == "true"

    def activate(self) -> bool:
        """Bring Chrome to the foreground."""
        script = '''
        tell application "Google Chrome"
            activate
        end tell
        '''
        run_applescript(script)
        return True

    def get_tabs(self) -> List[Dict[str, Any]]:
        """Get all open tabs across all windows.

        Returns:
            List of dicts with 'title', 'url', 'index', 'window_index', 'active'
        """
        script = '''
        tell application "Google Chrome"
            set output to ""
            set windowIndex to 1
            repeat with w in windows
                set tabIndex to 1
                set activeTabIndex to active tab index of w
                repeat with t in tabs of w
                    set isActive to (tabIndex = activeTabIndex)
                    set output to output & (title of t) & "|" & (URL of t) & "|" & tabIndex & "|" & windowIndex & "|" & isActive & "\\n"
                    set tabIndex to tabIndex + 1
                end repeat
                set windowIndex to windowIndex + 1
            end repeat
            return output
        end tell
        '''

        result = run_applescript(script)
        tabs = []

        for line in result.strip().split('\n'):
            if not line or '|' not in line:
                continue
            parts = line.split('|')
            if len(parts) >= 5:
                try:
                    tabs.append({
                        'title': parts[0],
                        'url': parts[1],
                        'index': int(parts[2]),
                        'window_index': int(parts[3]),
                        'active': parts[4].lower() == 'true',
                    })
                except (ValueError, IndexError):
                    continue

        return tabs

    def get_current_tab(self) -> Optional[Dict[str, Any]]:
        """Get the currently active tab.

        Returns:
            Dict with 'title', 'url', 'index', 'window_index', or None
        """
        script = '''
        tell application "Google Chrome"
            set t to active tab of front window
            set tabTitle to title of t
            set tabURL to URL of t
            set tabIndex to active tab index of front window
            return tabTitle & "|" & tabURL & "|" & tabIndex
        end tell
        '''

        result = run_applescript(script)
        if result and '|' in result:
            parts = result.split('|')
            if len(parts) >= 3:
                try:
                    return {
                        'title': parts[0],
                        'url': parts[1],
                        'index': int(parts[2]),
                        'window_index': 1,
                        'active': True,
                    }
                except (ValueError, IndexError):
                    pass
        return None

    def close_tab(self, index: int = None, title: str = None,
                  window_index: int = 1) -> Dict[str, Any]:
        """Close a tab by index or title match.

        Args:
            index: Tab index (1-based)
            title: Partial title match
            window_index: Window index (default: front window)

        Returns:
            Dict with 'success', 'message', optionally 'closed_tab'
        """
        if index is None and title is None:
            # Close current tab
            script = '''
            tell application "Google Chrome"
                set tabTitle to title of active tab of front window
                close active tab of front window
                return tabTitle
            end tell
            '''
            result = run_applescript(script)
            if result:
                return {'success': True, 'message': f'Closed tab: {result}', 'closed_tab': result}
            return {'success': False, 'message': 'Failed to close tab'}

        if title:
            # Find tab by title
            tabs = self.get_tabs()
            title_lower = title.lower()
            for tab in tabs:
                if title_lower in tab['title'].lower() or title_lower in tab['url'].lower():
                    index = tab['index']
                    window_index = tab['window_index']
                    break
            else:
                return {'success': False, 'message': f'No tab found matching: {title}'}

        script = f'''
        tell application "Google Chrome"
            set tabTitle to title of tab {index} of window {window_index}
            close tab {index} of window {window_index}
            return tabTitle
        end tell
        '''
        result = run_applescript(script)
        if result:
            return {'success': True, 'message': f'Closed tab: {result}', 'closed_tab': result}
        return {'success': False, 'message': 'Failed to close tab'}

    def switch_tab(self, index: int = None, title: str = None,
                   window_index: int = 1) -> Dict[str, Any]:
        """Switch to a tab by index or title match.

        Args:
            index: Tab index (1-based)
            title: Partial title match
            window_index: Window index (default: front window)

        Returns:
            Dict with 'success', 'message', optionally 'tab'
        """
        if title:
            # Find tab by title
            tabs = self.get_tabs()
            title_lower = title.lower()
            for tab in tabs:
                if title_lower in tab['title'].lower() or title_lower in tab['url'].lower():
                    index = tab['index']
                    window_index = tab['window_index']
                    break
            else:
                return {'success': False, 'message': f'No tab found matching: {title}'}

        if index is None:
            return {'success': False, 'message': 'No tab index or title specified'}

        script = f'''
        tell application "Google Chrome"
            set active tab index of window {window_index} to {index}
            set index of window {window_index} to 1
            set tabTitle to title of active tab of window {window_index}
            return tabTitle
        end tell
        '''
        result = run_applescript(script)
        if result:
            return {'success': True, 'message': f'Switched to: {result}', 'tab': result}
        return {'success': False, 'message': 'Failed to switch tab'}

    def new_tab(self, url: str = None) -> Dict[str, Any]:
        """Open a new tab, optionally navigating to a URL.

        Args:
            url: URL to navigate to (optional)

        Returns:
            Dict with 'success', 'message'
        """
        if url:
            script = f'''
            tell application "Google Chrome"
                tell front window
                    make new tab with properties {{URL:"{url}"}}
                end tell
            end tell
            '''
        else:
            script = '''
            tell application "Google Chrome"
                tell front window
                    make new tab
                end tell
            end tell
            '''

        run_applescript(script)
        return {'success': True, 'message': f'Opened new tab' + (f' with {url}' if url else '')}

    def navigate(self, url: str) -> bool:
        """Navigate the current tab to a URL.

        Args:
            url: URL to navigate to

        Returns:
            True if navigation was successful
        """
        # Ensure URL has a protocol
        if not url.startswith(('http://', 'https://', 'file://')):
            url = 'https://' + url

        script = f'''
        tell application "Google Chrome"
            set URL of active tab of front window to "{url}"
        end tell
        '''
        run_applescript(script)
        return True

    def get_url_bar(self) -> Optional[UIElement]:
        """Get the URL bar element.

        Returns:
            UIElement for the URL bar, or None if not found
        """
        script = '''
        tell application "System Events"
            tell process "Google Chrome"
                try
                    set urlBar to text field 1 of toolbar 1 of window 1
                    set urlPos to position of urlBar
                    set urlSize to size of urlBar
                    set urlValue to value of urlBar
                    return (item 1 of urlPos) & "|" & (item 2 of urlPos) & "|" & (item 1 of urlSize) & "|" & (item 2 of urlSize) & "|" & urlValue
                on error
                    return ""
                end try
            end tell
        end tell
        '''

        result = run_applescript(script)
        if result and '|' in result:
            parts = result.split('|')
            if len(parts) >= 4:
                try:
                    return UIElement(
                        name="URL Bar",
                        role="text field",
                        position=(int(parts[0]), int(parts[1])),
                        size=(int(parts[2]), int(parts[3])),
                        app="Google Chrome",
                        value=parts[4] if len(parts) > 4 else None
                    )
                except (ValueError, IndexError):
                    pass
        return None

    def execute_javascript(self, script: str) -> Optional[str]:
        """Execute JavaScript in the current tab.

        Args:
            script: JavaScript code to execute

        Returns:
            Result of the script, or None on error
        """
        # Escape the script for AppleScript
        escaped = script.replace('\\', '\\\\').replace('"', '\\"')

        applescript = f'''
        tell application "Google Chrome"
            execute front window's active tab javascript "{escaped}"
        end tell
        '''
        return run_applescript(applescript) or None


class SafariHandler:
    """Safari-specific accessibility handler.

    Provides methods for interacting with Safari tabs, windows, and navigation.
    """

    def __init__(self):
        """Initialize the Safari handler."""
        self.app_name = "Safari"

    def is_running(self) -> bool:
        """Check if Safari is running."""
        script = '''
        tell application "System Events"
            return (name of processes) contains "Safari"
        end tell
        '''
        result = run_applescript(script)
        return result.lower() == "true"

    def activate(self) -> bool:
        """Bring Safari to the foreground."""
        script = '''
        tell application "Safari"
            activate
        end tell
        '''
        run_applescript(script)
        return True

    def get_tabs(self) -> List[Dict[str, Any]]:
        """Get all open tabs across all windows.

        Returns:
            List of dicts with 'title', 'url', 'index', 'window_index'
        """
        script = '''
        tell application "Safari"
            set output to ""
            set windowIndex to 1
            repeat with w in windows
                set tabIndex to 1
                repeat with t in tabs of w
                    set output to output & (name of t) & "|" & (URL of t) & "|" & tabIndex & "|" & windowIndex & "\\n"
                    set tabIndex to tabIndex + 1
                end repeat
                set windowIndex to windowIndex + 1
            end repeat
            return output
        end tell
        '''

        result = run_applescript(script)
        tabs = []

        for line in result.strip().split('\n'):
            if not line or '|' not in line:
                continue
            parts = line.split('|')
            if len(parts) >= 4:
                try:
                    tabs.append({
                        'title': parts[0],
                        'url': parts[1],
                        'index': int(parts[2]),
                        'window_index': int(parts[3]),
                    })
                except (ValueError, IndexError):
                    continue

        return tabs

    def close_tab(self, index: int = None, title: str = None,
                  window_index: int = 1) -> Dict[str, Any]:
        """Close a tab by index or title match."""
        if index is None and title is None:
            script = '''
            tell application "Safari"
                set tabName to name of current tab of front window
                close current tab of front window
                return tabName
            end tell
            '''
            result = run_applescript(script)
            if result:
                return {'success': True, 'message': f'Closed tab: {result}', 'closed_tab': result}
            return {'success': False, 'message': 'Failed to close tab'}

        if title:
            tabs = self.get_tabs()
            title_lower = title.lower()
            for tab in tabs:
                if title_lower in tab['title'].lower() or title_lower in tab['url'].lower():
                    index = tab['index']
                    window_index = tab['window_index']
                    break
            else:
                return {'success': False, 'message': f'No tab found matching: {title}'}

        script = f'''
        tell application "Safari"
            set tabName to name of tab {index} of window {window_index}
            close tab {index} of window {window_index}
            return tabName
        end tell
        '''
        result = run_applescript(script)
        if result:
            return {'success': True, 'message': f'Closed tab: {result}', 'closed_tab': result}
        return {'success': False, 'message': 'Failed to close tab'}

    def switch_tab(self, index: int = None, title: str = None,
                   window_index: int = 1) -> Dict[str, Any]:
        """Switch to a tab by index or title match."""
        if title:
            tabs = self.get_tabs()
            title_lower = title.lower()
            for tab in tabs:
                if title_lower in tab['title'].lower() or title_lower in tab['url'].lower():
                    index = tab['index']
                    window_index = tab['window_index']
                    break
            else:
                return {'success': False, 'message': f'No tab found matching: {title}'}

        if index is None:
            return {'success': False, 'message': 'No tab index or title specified'}

        script = f'''
        tell application "Safari"
            set current tab of window {window_index} to tab {index} of window {window_index}
            set index of window {window_index} to 1
            set tabName to name of current tab of window {window_index}
            return tabName
        end tell
        '''
        result = run_applescript(script)
        if result:
            return {'success': True, 'message': f'Switched to: {result}', 'tab': result}
        return {'success': False, 'message': 'Failed to switch tab'}

    def navigate(self, url: str) -> bool:
        """Navigate the current tab to a URL."""
        if not url.startswith(('http://', 'https://', 'file://')):
            url = 'https://' + url

        script = f'''
        tell application "Safari"
            set URL of current tab of front window to "{url}"
        end tell
        '''
        run_applescript(script)
        return True


class FinderHandler:
    """Finder-specific accessibility handler.

    Provides methods for interacting with Finder windows and files.

    Example:
        finder = FinderHandler()
        files = finder.get_selected_files()
        finder.navigate_to("~/Documents")
    """

    def __init__(self):
        """Initialize the Finder handler."""
        self.app_name = "Finder"

    def activate(self) -> bool:
        """Bring Finder to the foreground."""
        script = '''
        tell application "Finder"
            activate
        end tell
        '''
        run_applescript(script)
        return True

    def get_selected_files(self) -> List[str]:
        """Get paths of currently selected files.

        Returns:
            List of file paths
        """
        script = '''
        tell application "Finder"
            set selectedItems to selection
            set output to ""
            repeat with item_ref in selectedItems
                set itemPath to POSIX path of (item_ref as alias)
                set output to output & itemPath & "\\n"
            end repeat
            return output
        end tell
        '''

        result = run_applescript(script)
        files = []
        for line in result.strip().split('\n'):
            if line:
                files.append(line)
        return files

    def get_current_folder(self) -> Optional[str]:
        """Get the current folder in the front Finder window.

        Returns:
            Path to the current folder, or None
        """
        script = '''
        tell application "Finder"
            try
                set currentFolder to target of front window
                return POSIX path of (currentFolder as alias)
            on error
                return ""
            end try
        end tell
        '''
        result = run_applescript(script)
        return result if result else None

    def navigate_to(self, path: str) -> bool:
        """Navigate the front Finder window to a folder.

        Args:
            path: Folder path (supports ~ for home directory)

        Returns:
            True if navigation was successful
        """
        # Expand home directory
        if path.startswith('~'):
            path = os.path.expanduser(path)

        # Make sure path is absolute
        path = os.path.abspath(path)

        script = f'''
        tell application "Finder"
            try
                set target of front window to POSIX file "{path}"
                return "success"
            on error
                -- No window open, create one
                open POSIX file "{path}"
                return "success"
            end try
        end tell
        '''
        result = run_applescript(script)
        return result == "success"

    def new_window(self, path: str = None) -> bool:
        """Open a new Finder window.

        Args:
            path: Optional path to open (defaults to home directory)

        Returns:
            True if window was created
        """
        if path:
            if path.startswith('~'):
                path = os.path.expanduser(path)
            path = os.path.abspath(path)
            script = f'''
            tell application "Finder"
                open POSIX file "{path}"
            end tell
            '''
        else:
            script = '''
            tell application "Finder"
                make new Finder window
            end tell
            '''
        run_applescript(script)
        return True

    def reveal_in_finder(self, path: str) -> bool:
        """Reveal a file or folder in Finder.

        Args:
            path: Path to reveal

        Returns:
            True if reveal was successful
        """
        if path.startswith('~'):
            path = os.path.expanduser(path)
        path = os.path.abspath(path)

        script = f'''
        tell application "Finder"
            reveal POSIX file "{path}"
            activate
        end tell
        '''
        run_applescript(script)
        return True

    def get_clipboard_files(self) -> List[str]:
        """Get file paths from clipboard (if files were copied).

        Returns:
            List of file paths from clipboard
        """
        script = '''
        try
            set clipboardContent to the clipboard as «class furl»
            return POSIX path of clipboardContent
        on error
            try
                set clipboardContent to the clipboard as list
                set output to ""
                repeat with item_ref in clipboardContent
                    try
                        set itemPath to POSIX path of (item_ref as alias)
                        set output to output & itemPath & "\\n"
                    end try
                end repeat
                return output
            on error
                return ""
            end try
        end try
        '''
        result = run_applescript(script)
        files = []
        for line in result.strip().split('\n'):
            if line:
                files.append(line)
        return files

    def create_folder(self, name: str, location: str = None) -> Optional[str]:
        """Create a new folder.

        Args:
            name: Name of the new folder
            location: Parent folder path (defaults to current folder)

        Returns:
            Path to the new folder, or None on error
        """
        if location is None:
            location = self.get_current_folder()
        if location is None:
            location = os.path.expanduser('~')

        if location.startswith('~'):
            location = os.path.expanduser(location)
        location = os.path.abspath(location)

        script = f'''
        tell application "Finder"
            try
                set newFolder to make new folder at POSIX file "{location}" with properties {{name:"{name}"}}
                return POSIX path of (newFolder as alias)
            on error errMsg
                return ""
            end try
        end tell
        '''
        result = run_applescript(script)
        return result if result else None

    def trash_file(self, path: str) -> bool:
        """Move a file to trash.

        Args:
            path: Path to the file to trash

        Returns:
            True if successful
        """
        if path.startswith('~'):
            path = os.path.expanduser(path)
        path = os.path.abspath(path)

        script = f'''
        tell application "Finder"
            try
                delete POSIX file "{path}"
                return "success"
            on error
                return "error"
            end try
        end tell
        '''
        result = run_applescript(script)
        return result == "success"


class SystemHandler:
    """System-wide accessibility handler.

    Provides methods for system-level interactions like dock, menu bar,
    notifications, and application management.

    Example:
        system = SystemHandler()
        app = system.get_active_app()
        system.show_notification("Hello!", "Message body")
    """

    def __init__(self):
        """Initialize the system handler."""
        self._finder = get_finder()

    def get_active_app(self) -> str:
        """Get the name of the frontmost application.

        Returns:
            Application name
        """
        return self._finder.get_active_app()

    def get_running_apps(self) -> List[str]:
        """Get list of running applications.

        Returns:
            List of application names
        """
        script = '''
        tell application "System Events"
            set appNames to name of every application process whose visible is true
            set output to ""
            repeat with appName in appNames
                set output to output & appName & "\\n"
            end repeat
            return output
        end tell
        '''
        result = run_applescript(script)
        apps = []
        for line in result.strip().split('\n'):
            if line:
                apps.append(line)
        return apps

    def get_dock_items(self) -> List[UIElement]:
        """Get all items in the Dock.

        Returns:
            List of UIElements representing dock items
        """
        return self._finder.get_dock_items()

    def find_dock_item(self, name: str) -> Optional[UIElement]:
        """Find a dock item by name.

        Args:
            name: Name to search for (case-insensitive, partial match)

        Returns:
            UIElement for the dock item, or None
        """
        return self._finder.find_in_dock(name)

    def click_dock_item(self, name: str) -> bool:
        """Click on a dock item.

        Args:
            name: Name of the dock item

        Returns:
            True if click was successful
        """
        from .actions import get_actions
        item = self.find_dock_item(name)
        if item:
            actions = get_actions()
            return actions.click(item)
        return False

    def get_menubar_items(self) -> List[UIElement]:
        """Get menu bar items of the frontmost application.

        Returns:
            List of UIElements representing menu bar items
        """
        return self._finder.get_menubar_items()

    def click_menu(self, menu_name: str, item_name: str = None) -> bool:
        """Click a menu or menu item.

        Args:
            menu_name: Name of the menu (e.g., "File")
            item_name: Name of the menu item (e.g., "Save"). If None, just opens menu.

        Returns:
            True if click was successful
        """
        app = self.get_active_app()
        if not app:
            return False

        if item_name:
            script = f'''
            tell application "System Events"
                tell process "{app}"
                    click menu bar item "{menu_name}" of menu bar 1
                    delay 0.2
                    click menu item "{item_name}" of menu "{menu_name}" of menu bar item "{menu_name}" of menu bar 1
                end tell
            end tell
            '''
        else:
            script = f'''
            tell application "System Events"
                tell process "{app}"
                    click menu bar item "{menu_name}" of menu bar 1
                end tell
            end tell
            '''

        run_applescript(script)
        return True

    def get_notification_center(self) -> Optional[UIElement]:
        """Get the notification center element.

        Returns:
            UIElement for notification center, or None
        """
        script = '''
        tell application "System Events"
            tell process "NotificationCenter"
                try
                    set nc to window 1
                    set ncPos to position of nc
                    set ncSize to size of nc
                    return (item 1 of ncPos) & "|" & (item 2 of ncPos) & "|" & (item 1 of ncSize) & "|" & (item 2 of ncSize)
                on error
                    return ""
                end try
            end tell
        end tell
        '''

        result = run_applescript(script)
        if result and '|' in result:
            parts = result.split('|')
            if len(parts) >= 4:
                try:
                    return UIElement(
                        name="Notification Center",
                        role="window",
                        position=(int(parts[0]), int(parts[1])),
                        size=(int(parts[2]), int(parts[3])),
                        app="NotificationCenter"
                    )
                except (ValueError, IndexError):
                    pass
        return None

    def show_notification(self, title: str, message: str = "",
                         sound: bool = False) -> bool:
        """Show a system notification.

        Args:
            title: Notification title
            message: Notification body text
            sound: Whether to play a sound

        Returns:
            True if notification was displayed
        """
        sound_str = ' sound name "default"' if sound else ''
        script = f'''
        display notification "{message}" with title "{title}"{sound_str}
        '''
        run_applescript(script)
        return True

    def show_dialog(self, message: str, title: str = "Aria",
                   buttons: List[str] = None) -> Optional[str]:
        """Show a dialog with buttons.

        Args:
            message: Dialog message
            title: Dialog title
            buttons: List of button names (default: ["OK"])

        Returns:
            The button that was clicked, or None if cancelled
        """
        if buttons is None:
            buttons = ["OK"]

        buttons_str = ', '.join(f'"{b}"' for b in buttons)
        script = f'''
        display dialog "{message}" with title "{title}" buttons {{{buttons_str}}} default button 1
        '''
        result = run_applescript(script)
        if result:
            # Parse "button returned:OK"
            match = re.search(r'button returned:(.+)', result)
            if match:
                return match.group(1)
        return None

    def open_url(self, url: str, browser: str = None) -> bool:
        """Open a URL in the default or specified browser.

        Args:
            url: URL to open
            browser: Optional browser name (e.g., "Safari", "Google Chrome")

        Returns:
            True if successful
        """
        if not url.startswith(('http://', 'https://', 'file://')):
            url = 'https://' + url

        if browser:
            script = f'''
            tell application "{browser}"
                activate
                open location "{url}"
            end tell
            '''
        else:
            script = f'open location "{url}"'

        run_applescript(script)
        return True

    def open_app(self, app_name: str) -> bool:
        """Open an application.

        Args:
            app_name: Name of the application

        Returns:
            True if successful
        """
        script = f'''
        tell application "{app_name}"
            activate
        end tell
        '''
        run_applescript(script)
        return True

    def quit_app(self, app_name: str) -> bool:
        """Quit an application.

        Args:
            app_name: Name of the application

        Returns:
            True if successful
        """
        script = f'''
        tell application "{app_name}"
            quit
        end tell
        '''
        run_applescript(script)
        return True

    def get_system_info(self) -> Dict[str, Any]:
        """Get basic system information.

        Returns:
            Dict with system info
        """
        script = '''
        set computerName to do shell script "scutil --get ComputerName"
        set osVersion to do shell script "sw_vers -productVersion"
        return computerName & "|" & osVersion
        '''
        result = run_applescript(script)
        parts = result.split('|') if result else ["", ""]

        return {
            'computer_name': parts[0] if len(parts) > 0 else "",
            'os_version': parts[1] if len(parts) > 1 else "",
        }

    def set_volume(self, level: int) -> bool:
        """Set the system volume.

        Args:
            level: Volume level (0-100)

        Returns:
            True if successful
        """
        level = max(0, min(100, level))
        # Convert 0-100 to 0-7 (Mac volume scale)
        mac_level = int(level * 7 / 100)
        script = f'set volume output volume {level}'
        run_applescript(script)
        return True

    def get_volume(self) -> int:
        """Get the current system volume.

        Returns:
            Volume level (0-100)
        """
        script = 'output volume of (get volume settings)'
        result = run_applescript(script)
        try:
            return int(result)
        except (ValueError, TypeError):
            return 0

    def toggle_mute(self) -> bool:
        """Toggle system mute.

        Returns:
            True if now muted, False if unmuted
        """
        script = '''
        set currentMute to output muted of (get volume settings)
        if currentMute then
            set volume without output muted
            return "false"
        else
            set volume with output muted
            return "true"
        end if
        '''
        result = run_applescript(script)
        return result == "true"

    def take_screenshot(self, path: str = None, region: tuple = None) -> Optional[str]:
        """Take a screenshot.

        Args:
            path: Path to save screenshot (default: Desktop)
            region: Optional (x, y, width, height) to capture

        Returns:
            Path to the screenshot file
        """
        import subprocess
        import datetime

        if path is None:
            desktop = os.path.expanduser('~/Desktop')
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            path = os.path.join(desktop, f'Screenshot_{timestamp}.png')

        if region:
            x, y, w, h = region
            cmd = ['screencapture', '-R', f'{x},{y},{w},{h}', path]
        else:
            cmd = ['screencapture', '-x', path]

        try:
            subprocess.run(cmd, check=True)
            return path
        except subprocess.CalledProcessError:
            return None


class TerminalHandler:
    """Terminal-specific accessibility handler.

    Provides methods for interacting with Terminal.app.
    """

    def __init__(self):
        """Initialize the Terminal handler."""
        self.app_name = "Terminal"

    def activate(self) -> bool:
        """Bring Terminal to the foreground."""
        script = '''
        tell application "Terminal"
            activate
        end tell
        '''
        run_applescript(script)
        return True

    def new_window(self, command: str = None) -> bool:
        """Open a new Terminal window.

        Args:
            command: Optional command to run

        Returns:
            True if successful
        """
        if command:
            script = f'''
            tell application "Terminal"
                do script "{command}"
                activate
            end tell
            '''
        else:
            script = '''
            tell application "Terminal"
                do script ""
                activate
            end tell
            '''
        run_applescript(script)
        return True

    def run_command(self, command: str) -> bool:
        """Run a command in the current Terminal window.

        Args:
            command: Command to run

        Returns:
            True if successful
        """
        # Escape special characters
        escaped = command.replace('\\', '\\\\').replace('"', '\\"')

        script = f'''
        tell application "Terminal"
            do script "{escaped}" in front window
        end tell
        '''
        run_applescript(script)
        return True

    def get_current_directory(self) -> Optional[str]:
        """Get the current working directory of the front Terminal window.

        Returns:
            Path to current directory, or None
        """
        script = '''
        tell application "Terminal"
            try
                set currentTTY to tty of selected tab of front window
                set pwd to do shell script "lsof -a -p $(lsof -t " & currentTTY & " | head -1) -d cwd -F n | grep '^n' | cut -c2-"
                return pwd
            on error
                return ""
            end try
        end tell
        '''
        result = run_applescript(script)
        return result if result else None
