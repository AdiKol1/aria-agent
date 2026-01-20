#!/bin/bash
# Launch Aria - Double-click this file to start

cd "/Users/adikol/Desktop/aria-agent"
source "/Users/adikol/Desktop/anthropic-quickstarts/aria-agent/venv/bin/activate"

echo "================================"
echo "   Starting Aria Agent v0.1"
echo "================================"
echo ""
echo "Aria will appear in your menubar."
echo "Say 'Hey Aria' or press ⌥ Space to activate."
echo ""
echo "Required permissions (System Settings > Privacy & Security):"
echo "  - Microphone"
echo "  - Screen Recording"
echo "  - Accessibility"
echo ""
echo "Press Ctrl+C to quit."
echo "================================"
echo ""

python -m aria.main
