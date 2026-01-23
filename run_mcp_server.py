#!/usr/bin/env python3
"""
Run the Aria MCP Server

This script starts the Aria MCP server, which exposes Aria's capabilities
(screen capture, computer control, memory, voice) to Claude Code.
"""

import sys
import os

# Add the project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aria.mcp_server import run_server

if __name__ == "__main__":
    run_server()
