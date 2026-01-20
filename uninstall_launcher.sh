#!/bin/bash
#
# Aria Launcher Uninstaller
# Removes the Aria menubar launcher
#

set -e

PLIST_NAME="com.aria.launcher.plist"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST_DEST="$LAUNCH_AGENTS_DIR/$PLIST_NAME"

echo "================================"
echo "  Aria Launcher Uninstaller"
echo "================================"
echo ""

# Stop the launcher
echo "Stopping Aria Launcher..."
launchctl unload "$PLIST_DEST" 2>/dev/null || true
pkill -f "aria_launcher.py" 2>/dev/null || true

# Remove the plist
if [ -f "$PLIST_DEST" ]; then
    echo "Removing Launch Agent..."
    rm "$PLIST_DEST"
fi

echo ""
echo "================================"
echo "  Uninstall Complete!"
echo "================================"
echo ""
echo "Aria Launcher has been removed."
echo "It will no longer start at login."
echo ""
echo "Note: Aria itself is still installed."
echo "You can still run it manually with: python -m aria.main"
echo ""
