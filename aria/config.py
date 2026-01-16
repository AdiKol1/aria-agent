"""
Configuration for Aria Agent
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PORCUPINE_ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")

# Paths
ARIA_HOME = Path.home() / ".aria"
ARIA_HOME.mkdir(exist_ok=True)

DATA_PATH = ARIA_HOME / "data"
DATA_PATH.mkdir(exist_ok=True)

MEMORY_DB_PATH = ARIA_HOME / "memory.db"
SCREENSHOTS_PATH = ARIA_HOME / "screenshots"
SCREENSHOTS_PATH.mkdir(exist_ok=True)

# Wake Word
WAKE_WORD = "hey aria"

# Privacy - Apps to ignore (won't capture screen when these are focused)
PRIVATE_APPS = [
    "1Password",
    "Keychain Access",
    "System Preferences",
    "System Settings",
]

# Screen Capture
CAPTURE_INTERVAL_MS = 1000  # How often to capture screen when active
MAX_SCREENSHOT_AGE_HOURS = 24  # Delete old screenshots after this

# Voice
VOICE_SAMPLE_RATE = 24000
VOICE_CHANNELS = 1

# OpenAI Realtime API
# When OPENAI_API_KEY is set, realtime voice mode is available
REALTIME_VOICE_ENABLED = bool(OPENAI_API_KEY)
REALTIME_VOICE_MODEL = "gpt-4o-realtime-preview-2024-12-17"
REALTIME_VOICE_VOICE = "alloy"  # Options: alloy, echo, fable, onyx, nova, shimmer
REALTIME_VAD_THRESHOLD = 0.5  # Voice activity detection threshold (0.0 to 1.0)
REALTIME_SILENCE_DURATION_MS = 500  # Duration of silence to detect end of speech

# Claude Models
CLAUDE_MODEL = "claude-sonnet-4-20250514"  # For complex reasoning
CLAUDE_MODEL_FAST = "claude-3-5-haiku-20241022"  # For quick actions
CLAUDE_MAX_TOKENS = 4096

# Speed settings
ACTION_DELAY_MS = 300  # Delay between actions (ms)
SCREENSHOT_MAX_WIDTH = 1280  # Smaller = faster (was 1920)
SCREENSHOT_QUALITY = 70  # JPEG quality (1-100)

# MCP Server configurations
# These are external MCP servers that Aria can connect to for additional capabilities
MCP_SERVERS = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", str(Path.home())],
        "enabled": True,
        "description": "File system operations (read, write, list) in home directory",
    },
    "fetch": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-fetch"],
        "enabled": True,
        "description": "HTTP fetch operations for web requests",
    },
}

# MCP settings
MCP_AUTO_START = False  # Whether to start MCP servers automatically on init
MCP_TIMEOUT_SECONDS = 30  # Timeout for MCP server operations

# Validate required keys
def validate_config():
    """Check that required API keys are set."""
    missing = []
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    # Porcupine is optional - can use open source wake word instead
    return missing
