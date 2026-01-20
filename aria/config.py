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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
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
WAKE_WORD = "aria"  # Just say "Aria" to activate

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

# Voice Latency Optimization
VOICE_SILENCE_DURATION = 0.5  # Shorter silence detection (was 0.7)
VOICE_MIN_AUDIO_LENGTH = 0.3  # Minimum audio length in seconds

# Gemini Live API (DISABLED - Using Claude Opus 4.5 for maximum intelligence)
# Note: Gemini voice mode uses Gemini's brain instead of Claude
# We want Claude Opus 4.5 as the brain for complex reasoning and computer use
GEMINI_VOICE_ENABLED = False  # Disabled to use Claude Opus 4.5 brain
GEMINI_VOICE_MODEL = "gemini-2.0-flash-exp"  # Reverting to working model
GEMINI_VOICE_VOICE = "Kore"  # Options: Puck, Charon, Kore, Fenrir, Aoede (Kore/Aoede are female)
GEMINI_SAMPLE_RATE = 16000  # Gemini Live requires 16kHz input
GEMINI_OUTPUT_SAMPLE_RATE = 24000  # Higher quality output

# FALLBACK SYSTEM CONFIGURATION
# IMMEDIATE fallback: Triggers on action keywords during speech - CAUSES GARBAGE, keep disabled
# END-OF-TURN fallback: Triggers when Gemini claims action without tool call - useful
DISABLE_IMMEDIATE_FALLBACK = True   # Disable: triggers on partial speech, causes garbage
DISABLE_CONFABULATION_FALLBACK = False  # Enable: catches when Gemini lies about doing actions

# Legacy flag (for backwards compatibility) - prefer the specific flags above
DISABLE_FALLBACK_SYSTEM = True  # Kept for any code still using it

# OpenAI Realtime API - Fast voice I/O with tool routing
# Uses GPT-4o for fast STT/TTS and simple actions (open app, click, type)
# Complex tasks can be routed to Claude via execute_task tool
REALTIME_VOICE_ENABLED = True  # Enabled for fast voice I/O - uses GPT-4o for simple actions, routes complex to Claude
REALTIME_VOICE_MODEL = "gpt-4o-realtime-preview-2024-12-17"
REALTIME_VOICE_VOICE = "alloy"  # Options: alloy, echo, fable, onyx, nova, shimmer
REALTIME_VAD_THRESHOLD = 0.5  # Voice activity detection threshold (0.0 to 1.0)
REALTIME_SILENCE_DURATION_MS = 500  # Duration of silence to detect end of speech

# Claude Models
CLAUDE_MODEL = "claude-opus-4-5-20251101"  # Opus 4.5 - Maximum intelligence for complex tasks
CLAUDE_MODEL_FAST = "claude-sonnet-4-20250514"  # Sonnet for quick actions
CLAUDE_MAX_TOKENS = 4096

# Smart Model Selection
# Use faster models for simple tasks, Opus for complex reasoning
CLAUDE_MODEL_SIMPLE = "claude-sonnet-4-20250514"  # For simple tasks (open app, click)
CLAUDE_MODEL_COMPLEX = "claude-opus-4-5-20251101"  # For complex reasoning
USE_SMART_MODEL_SELECTION = True  # Enable automatic model selection

# Prompt Caching (reduces TTFT significantly)
ENABLE_PROMPT_CACHING = True

# Speed settings
ACTION_DELAY_MS = 150  # Delay between actions (ms)
SCREENSHOT_MAX_WIDTH = 1280  # Reduced resolution for faster processing
SCREENSHOT_QUALITY = 95  # High JPEG quality for better accuracy

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
