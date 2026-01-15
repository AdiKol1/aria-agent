# Aria Agent - Claude Code Integration

## Overview

Aria is a voice-first desktop assistant that integrates with Claude Code. This enables:

1. **Voice Coding** - Say "Hey Aria, fix the bug in login.ts" and Claude Code executes it
2. **Desktop Superpowers** - Claude Code can see your screen and control your computer
3. **Shared Memory** - Both Aria and Claude Code share the same long-term memory

## Quick Start

### Running Aria (Voice Assistant)
```bash
cd /Users/adikol/Desktop/anthropic-quickstarts/aria-agent
source venv/bin/activate
python -m aria.main
```

### Connecting Claude Code to Aria

The Aria MCP server is configured in `~/.mcp.json` (global) and `.mcp.json` (project).

When you start Claude Code, it will prompt you to enable the Aria MCP server.
Select "Yes" to enable it.

The configuration is already set up at:
- Global: `~/.mcp.json`
- Project: `/Users/adikol/Desktop/anthropic-quickstarts/aria-agent/.mcp.json`

```json
{
  "mcpServers": {
    "aria": {
      "command": "python3",
      "args": ["/Users/adikol/Desktop/anthropic-quickstarts/aria-agent/run_mcp_server.py"]
    }
  }
}
```

## Available MCP Tools

When connected, Claude Code gains these Aria capabilities:

### Screen & Vision
- `capture_screen` - Get a screenshot of the current display
- `get_active_app` - Get the name of the frontmost application

### Computer Control
- `click(x, y)` - Click at coordinates
- `double_click(x, y)` - Double-click
- `scroll(amount)` - Scroll up/down
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

## Memory System

Memory is stored in `~/.aria/data/memory/` using ChromaDB. Both Aria and Claude Code share this memory.

### Memory Categories
- `preference` - User preferences ("prefers dark mode")
- `personal` - Personal info ("name is Adi")
- `work` - Work-related ("works at Anthropic")
- `habit` - Habits and patterns
- `project` - Project-specific info
- `other` - Everything else

### Example Usage in Claude Code
```
User: Remember that I prefer TypeScript over JavaScript
Claude: [Uses remember tool with fact="User prefers TypeScript over JavaScript", category="preference"]

User: What do you know about my preferences?
Claude: [Uses recall tool with query="preferences"]
```

## Voice Coding Mode

When you tell Aria something coding-related, it automatically delegates to Claude Code:

**Triggers:**
- "fix bug", "write code", "implement", "refactor"
- "edit file", "create file", "find file"
- "commit", "push", "git"
- "run tests", "build", "npm", "pip"
- "in the codebase", "in the code"

**Examples:**
- "Hey Aria, fix the TypeScript errors in the auth module"
- "Hey Aria, add a loading spinner to the dashboard"
- "Hey Aria, commit these changes with a good message"

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User                                  │
│              (Voice or Terminal)                            │
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

## Files

- `aria/main.py` - Menubar app entry point
- `aria/agent.py` - Core agent brain (Claude API + actions)
- `aria/memory.py` - ChromaDB long-term memory
- `aria/claude_bridge.py` - Delegates to Claude Code CLI
- `aria/mcp_server.py` - MCP server for Claude Code
- `aria/vision.py` - Screen capture
- `aria/control.py` - Computer control (pyautogui)
- `aria/voice.py` - Voice I/O (OpenAI Whisper + TTS)
- `run_mcp_server.py` - Script to run MCP server
