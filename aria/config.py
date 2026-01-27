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

# =============================================================================
# DAEMON CONFIGURATION (v4.0)
# =============================================================================
# Aria can run as a background daemon with REST API access

DAEMON_PORT = int(os.getenv("ARIA_DAEMON_PORT", "8420"))
DAEMON_HOST = os.getenv("ARIA_DAEMON_HOST", "127.0.0.1")
DAEMON_AUTO_START = os.getenv("ARIA_DAEMON_AUTO_START", "false").lower() == "true"

# =============================================================================
# PUSH NOTIFICATIONS (v4.0)
# =============================================================================
# Uses ntfy.sh for cross-platform push notifications

NTFY_SERVER = os.getenv("NTFY_SERVER", "https://ntfy.sh")
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "")  # User must configure their own topic
NTFY_TOKEN = os.getenv("NTFY_TOKEN", "")  # Optional: for authenticated topics

# =============================================================================
# TELEGRAM BRIDGE (v4.0)
# =============================================================================
# Access Aria from Telegram messaging

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN)

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

# =============================================================================
# VOICE SYSTEM v2.0: Moshi + Claude
# =============================================================================
# Moshi: Local voice I/O on Apple Silicon (160-200ms latency, free)
# Claude: Complex reasoning and computer control
# Local: Instant execution for simple commands (~50ms)

MOSHI_ENABLED = False  # v2.0 - disabled, using Pipecat v3.0 instead
MOSHI_MODEL = "kyutai/moshiko-mlx-q4"  # Quantized for Apple Silicon
MOSHI_SAMPLE_RATE = 24000
MOSHI_FRAME_SIZE_MS = 80

# =============================================================================
# VOICE SYSTEM v3.0: Pipecat (Deepgram + Claude + ElevenLabs)
# =============================================================================
# Production-ready voice pipeline with:
# - Deepgram STT: Real-time streaming transcription (~200ms)
# - Claude: Reasoning and tool execution
# - ElevenLabs: Natural voice synthesis
# - Silero VAD: Voice activity detection + interruption handling

PIPECAT_ENABLED = True  # v3.0 uses Pipecat pipeline
PIPECAT_FALLBACK_TO_LOCAL = True  # Fall back to traditional voice if Pipecat fails

# Deepgram STT Configuration
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_MODEL = "nova-2"  # Best accuracy/speed balance
DEEPGRAM_LANGUAGE = "en-US"

# ElevenLabs TTS Configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel - natural female voice
ELEVENLABS_MODEL = "eleven_turbo_v2_5"  # Low latency model

# VAD Configuration (Voice Activity Detection)
PIPECAT_VAD_STOP_SECS = 1.0  # Silence duration before end-of-turn (increased to prevent cutting off)
PIPECAT_ENABLE_INTERRUPTIONS = False  # Disabled to prevent echo feedback loop

# Pipeline Settings
PIPECAT_SAMPLE_RATE = 16000  # Deepgram optimal rate

# ========== DEPRECATED: Old Voice Systems (removed in v2.0) ==========
# These settings are kept for backwards compatibility but are not used.
# The old Gemini Live, OpenAI Realtime, and Qwen Hybrid systems have been
# replaced by Moshi + Claude architecture for better latency and cost.

# Gemini Live (REMOVED in v2.0)
GEMINI_VOICE_ENABLED = False  # Deprecated - use Moshi
GEMINI_VOICE_MODEL = ""
GEMINI_VOICE_VOICE = ""
GEMINI_SAMPLE_RATE = 16000
GEMINI_OUTPUT_SAMPLE_RATE = 24000

# Fallback system (REMOVED in v2.0)
DISABLE_IMMEDIATE_FALLBACK = True
DISABLE_CONFABULATION_FALLBACK = False
DISABLE_FALLBACK_SYSTEM = True

# OpenAI Realtime API (REMOVED in v2.0 - was $0.30/min, now free with Moshi)
REALTIME_VOICE_ENABLED = False  # Deprecated - use Moshi
REALTIME_VOICE_MODEL = ""
REALTIME_VOICE_VOICE = ""
REALTIME_VAD_THRESHOLD = 0.5
REALTIME_SILENCE_DURATION_MS = 500

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

# Qwen Hybrid (REMOVED in v2.0 - Replaced by Moshi + Claude)
# The Qwen model cache was deleted to free space for Moshi.
HYBRID_VOICE_ENABLED = False  # Deprecated - use Moshi
HYBRID_QWEN_MODEL = ""  # Model deleted
HYBRID_QWEN_QUANTIZATION = ""
HYBRID_USE_LOCAL_FOR_SIMPLE = True  # Still used by Moshi's TaskRouter
HYBRID_OLLAMA_MODEL = ""  # Deprecated

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
