# Aria Agent

An autonomous AI assistant for macOS that can see your screen, control your computer, and respond to voice commands.

## Features

- **Voice Activation**: Say "Hey Aria" to activate (or use ⌥ Space)
- **Screen Vision**: Aria can see what's on your screen using Claude's vision
- **Computer Control**: Click, type, scroll, open apps, use keyboard shortcuts
- **Voice Interaction**: Natural voice input and output using OpenAI

## Requirements

- macOS 12.0 or later
- Python 3.11+
- API Keys:
  - Anthropic API key (for Claude)
  - OpenAI API key (for voice)
  - Porcupine API key (optional, for wake word)

## Installation

```bash
# Clone the repo
cd aria-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys
```

## macOS Permissions

Aria requires the following permissions (grant in System Settings > Privacy & Security):

1. **Microphone** - For voice input
2. **Screen Recording** - For seeing your screen
3. **Accessibility** - For controlling your computer

## Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run Aria
python -m aria.main
```

Aria will appear in your menubar. You can:

- Say **"Hey Aria"** to activate (if Porcupine is configured)
- Press **⌥ Space** to activate manually
- Click the menubar icon for options

## Example Commands

```
"Hey Aria, what am I looking at?"
"Hey Aria, open Safari"
"Hey Aria, click the submit button"
"Hey Aria, type hello world"
"Hey Aria, scroll down"
"Hey Aria, what's on my calendar today?" (coming soon)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ARIA AGENT (v0.1)                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌───────────────┐  │
│   │  Menubar    │    │   Voice     │    │  Claude       │  │
│   │  App        │◄──►│   Engine    │◄──►│  Vision +     │  │
│   │  (rumps)    │    │  (OpenAI)   │    │  Reasoning    │  │
│   └─────────────┘    └─────────────┘    └───────────────┘  │
│         │                  │                   │            │
│         ▼                  ▼                   ▼            │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              macOS Integration                       │  │
│   │  Screen Capture │ pyautogui │ AppleScript           │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
aria-agent/
├── aria/
│   ├── __init__.py      # Package init
│   ├── main.py          # Menubar app entry point
│   ├── agent.py         # Core agent brain
│   ├── vision.py        # Screen capture + Claude vision
│   ├── control.py       # Computer control (mouse, keyboard)
│   ├── voice.py         # Voice input/output
│   ├── wake_word.py     # Wake word detection
│   └── config.py        # Configuration
├── requirements.txt
├── .env.example
└── README.md
```

## Roadmap

- [x] **v0.1**: Foundation (screen, control, voice)
- [ ] **v0.2**: Memory system (remember context)
- [ ] **v0.3**: Integrations (Gmail, Calendar, Notion)
- [ ] **v0.4**: Automation (record and replay workflows)
- [ ] **v1.0**: Full autonomy

## Privacy

- Screen captures are processed by Claude API but not stored
- No data is sent to any servers except OpenAI and Anthropic
- "Private apps" list prevents capture of sensitive apps
- All processing happens on your Mac

## License

MIT
