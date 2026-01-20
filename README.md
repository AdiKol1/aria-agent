# Aria Agent

An autonomous AI voice assistant for macOS that can see your screen, control your computer, and respond to voice commands using a hybrid OpenAI Realtime + Claude architecture.

## Features

- **Voice Activation**: Say "Hey Aria" or press ⌥ Space
- **Fast Voice I/O**: ~230ms latency via OpenAI Realtime API
- **Claude Brain**: Complex reasoning via Claude Opus 4.5
- **Screen Vision**: See and understand screen content
- **Computer Control**: Click, type, scroll, open apps
- **Long-term Memory**: Remember facts across sessions
- **Gesture Control**: Hands-free via webcam gestures
- **MCP Integration**: Works with Claude Code

## Quick Start

```bash
# Navigate to project
cd /Users/adikol/Desktop/aria-agent

# Activate virtual environment
source venv/bin/activate

# Run Aria
python -m aria.main
```

Aria will appear in your menubar (top-right). Activate by:
- Saying **"Hey Aria"**
- Pressing **⌥ Space** (Option + Space)
- Clicking the **Aria** menubar icon

## Requirements

- macOS 12.0 or later
- Python 3.9+ (3.11 recommended)
- API Keys (in `.env` file):
  - `ANTHROPIC_API_KEY` - For Claude (required)
  - `OPENAI_API_KEY` - For voice (required)
  - `GOOGLE_API_KEY` - For Gemini voice (optional)

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd aria-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys:
#   ANTHROPIC_API_KEY=sk-ant-...
#   OPENAI_API_KEY=sk-...
```

## macOS Permissions

Grant these in **System Settings → Privacy & Security**:

| Permission | Purpose | Required |
|------------|---------|----------|
| **Microphone** | Voice input | Yes |
| **Screen Recording** | Screen capture | Yes |
| **Accessibility** | Mouse/keyboard control | Yes |
| **Camera** | Gesture recognition | Optional |

## Configuration

Key settings in `aria/config.py`:

```python
# Voice Mode (MUST be True for fast voice)
REALTIME_VOICE_ENABLED = True

# Models
CLAUDE_MODEL = "claude-sonnet-4-20250514"      # Complex tasks
CLAUDE_MODEL_FAST = "claude-haiku-3-5-20241022" # Fast responses
```

## Example Commands

```
"Hey Aria, what am I looking at?"
"Hey Aria, open Chrome"
"Hey Aria, click the submit button"
"Hey Aria, type hello world"
"Hey Aria, scroll down"
"Hey Aria, remember that I prefer dark mode"
"Hey Aria, what do you know about me?"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ARIA AGENT                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌─────────────────┐    ┌────────────┐ │
│  │   Menubar    │    │  OpenAI         │    │   Claude   │ │
│  │   App        │◄──►│  Realtime API   │◄──►│   Brain    │ │
│  │   (rumps)    │    │  (Fast Voice)   │    │ (Reasoning)│ │
│  └──────────────┘    └─────────────────┘    └────────────┘ │
│         │                    │                    │         │
│         ▼                    ▼                    ▼         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              macOS Integration                        │  │
│  │  Screen Capture │ pyautogui │ Memory │ Gestures      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Voice Flow

```
User Voice → OpenAI Realtime (STT) → Task Router
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
              Simple Tasks          Computer Tasks        Complex Tasks
            (Realtime handles)    (Claude Computer Use)  (Claude Opus 4.5)
                    │                     │                     │
                    └─────────────────────┴─────────────────────┘
                                          │
                                          ▼
                              OpenAI Realtime (TTS) → User
```

## Project Structure

```
aria-agent/
├── aria/
│   ├── main.py              # Menubar app (lazy loads Gemini)
│   ├── agent.py             # Core brain (lazy loads gestures)
│   ├── config.py            # Configuration
│   ├── vision.py            # Screen capture
│   ├── control.py           # Mouse/keyboard control
│   ├── voice.py             # Traditional voice I/O
│   ├── realtime_voice.py    # OpenAI Realtime API
│   ├── gemini_voice.py      # Google Gemini Live API
│   ├── memory.py            # ChromaDB long-term memory
│   ├── mcp_server.py        # MCP server (lazy loading)
│   ├── gestures.py          # MediaPipe gesture recognition
│   ├── intent.py            # Intent understanding
│   ├── planner.py           # Task planning
│   ├── learning.py          # Learning from outcomes
│   └── claude_computer_use.py # Computer Use agent
├── .mcp.json                # MCP server config
├── requirements.txt
├── .env                     # API keys (create from .env.example)
├── CLAUDE.md               # Claude Code integration guide
└── README.md
```

## Troubleshooting

### Aria Won't Start

```bash
# Kill any existing processes
pkill -9 -f "aria"

# Try starting again
python -m aria.main
```

**First startup is slow** (~30 seconds) - this is normal due to SDK initialization.

### Voice Goes Silent

The Realtime API connection can timeout after ~5 min of silence.

**Fix**: Say "Hey Aria" or press ⌥ Space to reconnect.

### MCP Server Won't Connect

Check `.mcp.json` has correct path:
```json
{
  "mcpServers": {
    "aria": {
      "command": "/Users/adikol/Desktop/aria-agent/venv/bin/python",
      "args": ["/Users/adikol/Desktop/aria-agent/run_mcp_server.py"]
    }
  }
}
```

### Gestures Not Working

1. Grant camera permission: **System Settings → Privacy & Security → Camera → Terminal**
2. Or set: `export OPENCV_AVFOUNDATION_SKIP_AUTH=1`

### ⌥ Space Opens ChatGPT

Disable ChatGPT's hotkey or use "Hey Aria" / menubar instead.

## Performance Notes

### Lazy Loading

Heavy modules are loaded on-demand to ensure fast startup:

- **Gestures** (MediaPipe): Loaded only when `enable_gestures()` called
- **Gemini Voice** (Google SDK): Loaded only if Gemini mode selected
- **MCP Server**: Uses `@property` decorators for lazy init

### First vs Subsequent Starts

| Startup | Time | Reason |
|---------|------|--------|
| First | ~30s | Anthropic SDK cold start |
| Subsequent | ~2s | Cached imports |

## Known Issues

1. **Realtime API Timeout**: Connection drops after ~5 min silence
2. **MediaPipe First Load**: ~3 min first time (model download)
3. **Python 3.9 Warnings**: Google SDK deprecation warnings (safe to ignore)

## Gesture Controls

| Gesture | Action |
|---------|--------|
| 👍 Thumbs up | Confirm |
| 👎 Thumbs down | Cancel |
| ✋ Open palm | Stop |
| ✊ Closed fist | Pause |
| ☝️ Pointing up | Select |

Enable with:
```python
from aria.agent import get_agent
agent = get_agent()
agent.enable_gestures()
```

## MCP Tools (Claude Code Integration)

When connected to Claude Code, these tools are available:

- `capture_screen` - Screenshot
- `get_active_app` - Current app name
- `click(x, y)` - Click at coordinates
- `type_text(text)` - Type text
- `open_app(app)` - Open application
- `speak(text)` - Text-to-speech
- `remember(fact)` - Store to memory
- `recall(query)` - Search memory

See [CLAUDE.md](CLAUDE.md) for full integration guide.

## Roadmap

- [x] **v0.1**: Foundation (screen, control, voice)
- [x] **v1.0**: Memory system
- [x] **v2.0**: Intelligence layer (intent, planning, learning)
- [x] **v2.5**: Gesture recognition, performance optimizations
- [x] **v3.0**: Ambient intelligence (worlds, entities, goals)
- [ ] **v4.0**: Auto-reconnect for Realtime API, multi-modal input

## Privacy

- Screen captures processed by Claude API, not stored
- No data sent except to OpenAI/Anthropic APIs
- Memory stored locally in `~/.aria/data/`
- All processing on your Mac

## License

MIT
