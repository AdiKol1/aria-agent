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

# Claude
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKENS = 4096

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
