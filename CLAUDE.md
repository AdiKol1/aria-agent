# Aria Agent - Claude Code Integration

## Overview

Aria is a voice-first desktop assistant that integrates with Claude Code. This enables:

1. **Voice Coding** - Say "Hey Aria, fix the bug in login.ts" and Claude Code executes it
2. **Desktop Superpowers** - Claude Code can see your screen and control your computer
3. **Shared Memory** - Both Aria and Claude Code share the same long-term memory
4. **Gesture Control** - Hands-free confirmation via webcam gestures

## Quick Start

### Running Aria (Voice Assistant)
```bash
cd /Users/adikol/Desktop/aria-agent
source venv/bin/activate
python -m aria.main
```

Or use the Launch command:
```bash
./Launch\ Aria.command
```

### Connecting Claude Code to Aria

The Aria MCP server is configured in `.mcp.json` (project-level).

When you start Claude Code, it will prompt you to enable the Aria MCP server.
Select "Yes" to enable it.

**Project MCP Configuration** (`.mcp.json`):
```json
{
  "mcpServers": {
    "aria": {
      "command": "/Users/adikol/Desktop/aria-agent/venv/bin/python",
      "args": ["/Users/adikol/Desktop/aria-agent/run_mcp_server.py"],
      "env": {
        "PYTHONPATH": "/Users/adikol/Desktop/aria-agent"
      }
    }
  }
}
```

## Voice Architecture

Aria uses a **Hybrid Voice Architecture** for optimal performance:

### OpenAI Realtime API (Primary - Fast)
- **Latency**: ~230ms for voice I/O
- **Used for**: Speech-to-text, text-to-speech, simple actions
- **Config**: `REALTIME_VOICE_ENABLED=True` in `aria/config.py`

### Claude Brain (Intelligence)
- **Used for**: Complex reasoning, computer control, multi-step tasks
- **Model**: Claude Opus 4.5 for complex tasks, Claude Haiku for fast responses

### Traditional Mode (Fallback)
- Sequential: Whisper STT → Claude → TTS
- Higher latency but works without Realtime API

## Available MCP Tools

When connected, Claude Code gains these Aria capabilities:

### Screen & Vision
- `capture_screen` - Get a screenshot of the current display
- `get_active_app` - Get the name of the frontmost application
- `get_mouse_position` - Get current mouse coordinates

### Computer Control
- `click(x, y)` - Click at coordinates
- `double_click(x, y)` - Double-click
- `scroll(amount)` - Scroll up/down (positive=up, negative=down)
- `type_text(text)` - Type text (uses clipboard for reliability)
- `press_key(key)` - Press a key (enter, tab, escape, etc.)
- `hotkey(keys)` - Keyboard shortcut (e.g., ["command", "c"])
- `open_app(app)` - Open an application
- `open_url(url)` - Open a URL in browser

### Memory
- `remember(fact, category)` - Store a fact in long-term memory
- `recall(query)` - Search memory for relevant information
- `list_all_memories()` - List everything in memory

### Voice
- `speak(text)` - Speak text aloud (when Aria is running)

### Skills
- `list_skills()` - List all available Aria skills
- `run_skill(skill_name, input)` - Execute a skill by name

### Ambient Intelligence
- `get_briefing()` - Get status updates across all worlds
- `list_worlds()` - List monitored domains
- `create_world(name, description)` - Create a new monitoring domain
- `add_entity(world_id, name, type)` - Track a person/company/topic
- `add_goal(world_id, description)` - Add a goal to track
- `get_insights()` - Get pending insights that need attention

## Gesture Recognition

Aria supports hands-free control via webcam gestures:

| Gesture | Action |
|---------|--------|
| 👍 Thumbs up | Confirm |
| 👎 Thumbs down | Cancel |
| ✋ Open palm | Stop |
| ✊ Closed fist | Pause |
| ☝️ Pointing up | Select |

### Enabling Gestures
```python
from aria.agent import get_agent
agent = get_agent()
agent.enable_gestures()  # Requires camera permission
```

### Camera Permissions
Gestures require camera access. Grant permission in:
**System Settings → Privacy & Security → Camera → Terminal**

Or set environment variable to skip auth prompt:
```bash
export OPENCV_AVFOUNDATION_SKIP_AUTH=1
```

## Performance Optimizations

### Lazy Loading (Critical for Fast Startup)

Heavy modules are loaded lazily to ensure fast startup:

1. **Gestures** (`aria/agent.py`):
   - MediaPipe takes ~3 minutes to import first time
   - Loaded only when `enable_gestures()` is called

2. **Gemini Voice** (`aria/main.py`):
   - Google SDK is slow to import
   - Loaded only when Gemini voice mode is selected

3. **MCP Server** (`aria/mcp_server.py`):
   - All components use lazy initialization via `@property`
   - Server starts in ~0.02 seconds

### Configuration (`aria/config.py`)

```python
# Voice Mode - MUST be True for fast voice
REALTIME_VOICE_ENABLED = True

# Model Selection
CLAUDE_MODEL = "claude-sonnet-4-20250514"  # Complex tasks
CLAUDE_MODEL_FAST = "claude-haiku-3-5-20241022"  # Simple tasks
```

## Troubleshooting

### Aria Won't Start / Hangs on Import

**Symptom**: Process starts but no menubar icon appears

**Causes & Fixes**:
1. **Multiple processes running**: Kill all with `pkill -9 -f "aria"`
2. **Import hanging**: Check if mediapipe/google SDK is loading (can take minutes first time)
3. **SDK import error**: Run `python -c "import anthropic"` to test

### Voice Goes Silent / Stops Responding

**Symptom**: Aria stops responding after working for a while

**Cause**: OpenAI Realtime API connection timeout

**Fixes**:
1. Say "Hey Aria" or press ⌥ Space to reconnect
2. Restart Aria: `pkill -9 -f "aria.main" && python -m aria.main`

### MCP Server Won't Connect

**Symptom**: Claude Code shows "Failed to connect to aria"

**Fixes**:
1. Check path in `.mcp.json` points to correct location
2. Ensure venv Python is used: `/Users/adikol/Desktop/aria-agent/venv/bin/python`
3. Test manually: `python run_mcp_server.py`

### Camera Not Working for Gestures

**Symptom**: "Failed to open camera for gesture recognition"

**Fixes**:
1. Grant camera permission to Terminal in System Settings
2. Or set: `export OPENCV_AVFOUNDATION_SKIP_AUTH=1`

### Slow First Startup

**Symptom**: First startup takes 30+ seconds

**Cause**: Anthropic SDK cold start, MediaPipe model download

**This is normal** - subsequent startups are fast due to lazy loading.

## macOS Permissions Required

Grant these in **System Settings → Privacy & Security**:

| Permission | Purpose |
|------------|---------|
| **Microphone** | Voice input |
| **Screen Recording** | Screen capture |
| **Accessibility** | Computer control (mouse/keyboard) |
| **Camera** | Gesture recognition (optional) |

## Files

### Core
- `aria/main.py` - Menubar app entry point (lazy loads Gemini)
- `aria/agent.py` - Core agent brain (lazy loads gestures)
- `aria/memory.py` - ChromaDB long-term memory
- `aria/mcp_server.py` - MCP server for Claude Code (lazy loading)
- `aria/vision.py` - Screen capture
- `aria/control.py` - Computer control (pyautogui)
- `aria/voice.py` - Voice I/O (OpenAI Whisper + TTS)

### Voice Systems
- `aria/realtime_voice.py` - OpenAI Realtime API (fast, primary)
- `aria/gemini_voice.py` - Google Gemini Live API (alternative)
- `aria/hybrid_voice.py` - Hybrid voice routing

### Intelligence (v2.0)
- `aria/intent.py` - Intent understanding with memory
- `aria/planner.py` - Task planning and decomposition
- `aria/learning.py` - Learning from outcomes
- `aria/clarification.py` - Smart question asking
- `aria/gestures.py` - MediaPipe gesture recognition

### Skills System
- `aria/skills/` - Skills module
- `aria/skills/base.py` - Skill and SkillResult classes
- `aria/skills/registry.py` - @skill decorator and registry
- `aria/skills/loader.py` - Loads markdown and Python skills

### Computer Use
- `aria/claude_computer_use.py` - Claude Computer Use agent
- `aria/action_executor.py` - Vision-guided action execution

### Entry Points
- `run_mcp_server.py` - Script to run MCP server
- `Launch Aria.command` - Double-click to start Aria

### Ambient Intelligence (v3.0)
- `aria/ambient/` - Ambient intelligence system
- `aria/ambient/models.py` - Core data models
- `aria/ambient/world_manager.py` - World CRUD
- `aria/ambient/loop.py` - Main processing loop

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User                                  │
│         (Voice / Gestures / Terminal)                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌───────────────┐           ┌───────────────┐
│     Aria      │◄─────────►│  Claude Code  │
│ (Voice Agent) │  shared   │   (CLI Tool)  │
│               │  memory   │               │
└───────┬───────┘           └───────┬───────┘
        │                           │
        │         MCP Server        │
        └──────────►┌───────◄───────┘
                    │ Aria  │
                    │  MCP  │
                    └───┬───┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ Screen  │    │Computer │    │  Voice  │
   │ Capture │    │ Control │    │  I/O    │
   └─────────┘    └─────────┘    └─────────┘
```

## Voice Mode Selection

```
┌─────────────────────────────────────────────────────────────┐
│                    Voice Input                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
            ┌─────────────────┐
            │ Mode Selection  │
            └────────┬────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
    ▼                ▼                ▼
┌────────┐    ┌────────────┐    ┌──────────┐
│ Gemini │    │  Realtime  │    │Traditional│
│  Live  │    │   (Fast)   │    │ (Fallback)│
│  API   │    │  Primary   │    │           │
└────────┘    └────────────┘    └──────────┘
                     │
                     ▼
            ┌─────────────────┐
            │  Claude Brain   │
            │ (Complex Tasks) │
            └─────────────────┘
```

## Testing Checklist

After making changes, verify:

- [ ] `python -m aria.main` starts and shows menubar icon
- [ ] Voice responds to "Hey Aria" or ⌥ Space
- [ ] MCP tools work: `capture_screen`, `open_app`, `speak`
- [ ] Memory works: `remember`, `recall`
- [ ] Gestures work (if camera available)

## Known Issues

1. **Realtime API Timeout**: Connection drops after ~5 min of silence. Say "Hey Aria" to reconnect.
2. **MediaPipe First Load**: Takes ~3 minutes first time (model download).
3. **Python 3.9 Warnings**: Google SDK shows deprecation warnings (safe to ignore).
4. **ChatGPT Hotkey Conflict**: If ⌥ Space opens ChatGPT, disable its hotkey or use menubar/wake word.

## Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key

# Optional
GOOGLE_API_KEY=your_key  # For Gemini voice
REALTIME_VOICE_ENABLED=true  # Fast voice mode
```

## Version History

- **v0.1**: Initial release with voice control
- **v2.0**: Added intelligence layer (intent, planning, learning)
- **v3.0**: Added ambient intelligence and gesture recognition
- **Current**: Performance optimizations (lazy loading), stability fixes
