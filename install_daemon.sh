#!/bin/bash
# Install Aria Daemon as a macOS service
# This script sets up Aria to run as a background service via launchd

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST_SRC="$SCRIPT_DIR/com.aria.daemon.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.aria.daemon.plist"

echo "========================================"
echo "Aria Daemon Installer"
echo "========================================"
echo ""

# Check if plist source exists
if [ ! -f "$PLIST_SRC" ]; then
    echo "Error: com.aria.daemon.plist not found in $SCRIPT_DIR"
    exit 1
fi

# Check if venv exists
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Error: Virtual environment not found. Run:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Create LaunchAgents directory if it doesn't exist
mkdir -p "$HOME/Library/LaunchAgents"

# Stop existing daemon if running
if launchctl list | grep -q "com.aria.daemon"; then
    echo "Stopping existing Aria daemon..."
    launchctl unload "$PLIST_DST" 2>/dev/null || true
fi

# Copy plist
echo "Installing launchd service..."
cp "$PLIST_SRC" "$PLIST_DST"

# Create log directory
mkdir -p "$HOME/.aria"

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "To start the daemon now:"
echo "  launchctl load $PLIST_DST"
echo ""
echo "To stop the daemon:"
echo "  launchctl unload $PLIST_DST"
echo ""
echo "To check status:"
echo "  launchctl list | grep aria"
echo "  curl http://localhost:8420/api/v1/status"
echo ""
echo "To view logs:"
echo "  tail -f ~/.aria/daemon.log"
echo ""
echo "The daemon will auto-start on login."
echo ""

# Ask if user wants to start now
read -p "Start the daemon now? [Y/n] " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo "Starting Aria daemon..."
    launchctl load "$PLIST_DST"
    sleep 2

    # Check if running
    if curl -s http://localhost:8420/api/v1/health > /dev/null 2>&1; then
        echo "Aria daemon is running!"
        echo ""
        echo "API: http://localhost:8420"
        echo "Status: curl http://localhost:8420/api/v1/status"
    else
        echo "Daemon started but API not responding yet."
        echo "Check logs: tail -f ~/.aria/daemon.log"
    fi
fi
