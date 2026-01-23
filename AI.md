# AI.md — Start Here

**Read this file first. It tells you everything you need to know.**

## Quick Context

**Project**: Aria Agent - Autonomous macOS Desktop Assistant
**Stack**: Python 3.11+ | Claude (vision + reasoning) | OpenAI (voice) | rumps (menubar)
**Phase**: v0.1 - Foundation

## Before Coding

```bash
# Create venv and install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Copy env and add keys
cp .env.example .env

# Run
python -m aria.main
```

## Quality Gates

```bash
# Type check
pyright aria/

# Run (manual test)
python -m aria.main
```

## Key Rules

1. **Test on real Mac** — Requires permissions (mic, screen, accessibility)
2. **Privacy first** — Don't capture sensitive apps
3. **Voice UX** — Keep responses concise for speech
4. **Safety** — Confirm destructive actions

## Architecture

```
Wake Word → Voice Input → Claude (+ Screenshot) → Actions → Voice Output
```

## Key Files

| File | Purpose |
|------|---------|
| `aria/main.py` | Menubar app entry point |
| `aria/agent.py` | Core brain (Claude reasoning) |
| `aria/vision.py` | Screen capture + analysis |
| `aria/control.py` | Mouse, keyboard, app control |
| `aria/voice.py` | Speech-to-text, text-to-speech |
| `aria/wake_word.py` | "Hey Aria" detection |

## Current Phase (v0.1)

- [x] Menubar app
- [x] Screen capture + Claude vision
- [x] Computer control (click, type, scroll)
- [x] Voice input/output
- [ ] Wake word (needs Porcupine key)
- [ ] End-to-end testing

## Next Phase (v0.2)

- Memory system (SQLite + embeddings)
- Conversation persistence
- User preference learning

---

*Run `python -m aria.main` to test.*
