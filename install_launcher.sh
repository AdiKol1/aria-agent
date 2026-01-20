#!/bin/bash
#
# Aria Launcher Installer
# Installs the Aria menubar launcher to run at login
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST_NAME="com.aria.launcher.plist"
PLIST_SOURCE="$SCRIPT_DIR/$PLIST_NAME"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST_DEST="$LAUNCH_AGENTS_DIR/$PLIST_NAME"

echo "================================"
echo "  Aria Launcher Installer"
echo "================================"
echo ""

# Create LaunchAgents directory if it doesn't exist
if [ ! -d "$LAUNCH_AGENTS_DIR" ]; then
    echo "Creating LaunchAgents directory..."
    mkdir -p "$LAUNCH_AGENTS_DIR"
fi

# Stop existing launcher if running
echo "Stopping existing launcher (if any)..."
launchctl unload "$PLIST_DEST" 2>/dev/null || true
pkill -f "aria_launcher.py" 2>/dev/null || true

# Copy plist file
echo "Installing Launch Agent..."
cp "$PLIST_SOURCE" "$PLIST_DEST"

# Make the launcher script executable
chmod +x "$SCRIPT_DIR/aria_launcher.py"

# Load the Launch Agent
echo "Starting Aria Launcher..."
launchctl load "$PLIST_DEST"

echo ""
echo "================================"
echo "  Installation Complete!"
echo "================================"
echo ""
echo "Aria Launcher is now:"
echo "  - Running in your menu bar"
echo "  - Set to start automatically at login"
echo ""
echo "Click 'Aria' in the menu bar to start/stop Aria."
echo ""
echo "To uninstall, run: ./uninstall_launcher.sh"
echo ""
