# Aria Capability Research

> This document captures insights from analyzing agent repos to inform Aria's skill system design.
> Last updated: 2026-01-15

## IMPORTANT: Official Anthropic Skills Spec

Anthropic has an official Agent Skills specification at https://agentskills.io/specification

**Key Points:**
- Skills are folders with `SKILL.md` files
- Only `name` and `description` are required fields
- Optional directories: `scripts/`, `references/`, `assets/`
- Name format: lowercase, hyphens only, 1-64 chars
- See https://github.com/anthropics/skills for examples

**Aria Implementation:**
- Updated `aria/skills/loader.py` to support both formats
- Folder format: `skill-name/SKILL.md` (official)
- Flat format: `skill-name.md` (legacy, still supported)

## Research Goals
- Understand best patterns for agent capabilities/skills
- Identify tools and integrations worth adopting
- Learn from multi-agent coordination approaches
- Build foundation for Aria to self-discover new capabilities

---

## Executive Summary

After analyzing 11 major agent repositories, here are the **top patterns** to adopt for Aria:

### 🏆 Top 6 Patterns to Implement

| Pattern | Source | Why It Matters |
|---------|--------|----------------|
| **Markdown Skills with YAML** | Superpowers | Self-contained, human-readable skill files |
| **Registry-based Skills** | Browser-Use, Swarm | Decorator pattern for code-based skills |
| **Specialist Agent Handoffs** | Swarm, AutoGen, MetaGPT | Route tasks to domain experts |
| **MCP Tool Integration** | MCP Servers | 50+ ready-to-use tool servers |
| **Hook System** | Superpowers | Lifecycle events for context injection |
| **OpenAI Realtime Voice** | Realtime API | Sub-second voice latency |

### 🎯 Priority Implementation Order

1. **Markdown Skills System** - YAML frontmatter skills like Superpowers (perfect for voice!)
2. **Hook System** - Session lifecycle with context injection
3. **MCP Integration** - filesystem, memory, fetch servers
4. **Specialist Agents** - File, Browser, System, Memory agents
5. **Skill Shadowing** - User customization without forking
6. **Realtime Voice** - Upgrade from Whisper to native speech-to-speech

---

## Repos Analyzed

### 1. eigent-ai/eigent ✅
**Type:** Multi-agent desktop app

**Key Patterns:**
- Workforce architecture with specialized agents (coordinator, developer, search, document, multi-modal)
- Dynamic tool activation/deactivation at runtime
- MCP integration for extensibility
- Task decomposition with streaming progress
- Pause/resume for human-in-the-loop

**Adoptable Ideas:**
- [x] Specialized agent roles for different task types
- [x] Dynamic skill/tool activation
- [x] Task decomposition with progress streaming
- [x] Human confirmation for critical actions

---

### 2. OpenAdaptAI/OpenAdapt ✅
**Type:** AI-first process automation

**Key Patterns:**
- **Recording System**: Captures screenshots, mouse, keyboard, window state as structured data
- **Event Aggregation Pipeline**: Normalizes events (merge mouse moves, consolidate keypresses)
- **Strategy Pattern**: Pluggable replay strategies (Naive, Stateful, Visual, Vanilla)
- **Adapter/Driver Abstraction**: Model-agnostic LLM interface with fallback chain
- **Multi-Strategy Element Detection**: EXACT → STABLE → XPATH → AX_NAME → ATTRIBUTE

**Adoptable Ideas:**
- [ ] Event queue architecture for parallel capture
- [ ] Strategy pattern for action handlers
- [ ] Fallback chain for AI providers
- [ ] Visual grounding (voice + screenshot = targeted action)
- [ ] Demonstration learning for teaching complex workflows

---

### 3. microsoft/autogen ✅
**Type:** Multi-agent conversation framework

**Key Patterns:**
- **Three-Tier Architecture**: Core API → AgentChat API → Extensions API
- **Agent Types**: AssistantAgent, UserProxyAgent, CodeExecutorAgent, MultimodalWebSurfer, FileSurfer
- **Conversation Patterns**: RoundRobin, SelectorGroupChat, Swarm, Sequential, Concurrent
- **11 Termination Conditions**: MaxMessage, TextMention, Timeout, Handoff, External, etc.
- **Magentic-One**: Pre-built team with Orchestrator, WebSurfer, FileSurfer, Coder, Terminal

**Adoptable Ideas:**
- [ ] SelectorGroupChat for routing voice commands to specialists
- [ ] Handoff pattern for complex multi-domain tasks
- [ ] Termination conditions ("Hey Aria, stop")
- [ ] Tool wrapping with automatic schema generation

---

### 4. langchain-ai/langgraph ✅
**Type:** Stateful agent workflows

**Key Patterns:**
- **Graph Primitives**: Nodes (actions) + Edges (flow) + State (memory)
- **Super-step Execution**: Message-passing inspired by Google Pregel
- **Cycles**: Retry loops, iterative refinement
- **Conditional Branching**: Dynamic routing based on state
- **Checkpointing**: Persist state at every step, resume from failures
- **Streaming Modes**: values, updates, messages, custom

**State Schema with Reducers:**
```python
class AriaState(TypedDict):
    messages: Annotated[list, add_messages]  # Append
    current_screen: str  # Overwrite
    action_history: Annotated[list, add]  # Accumulate
```

**Adoptable Ideas:**
- [ ] Graph-based workflow modeling
- [ ] Checkpointing for crash recovery
- [ ] Thread-based session isolation
- [ ] Streaming for real-time voice output
- [ ] Human-in-the-loop via `interrupt()`

---

### 5. openai/swarm ✅
**Type:** Lightweight multi-agent orchestration

**Key Patterns:**
- **Minimal Primitives**: Only Agents and Handoffs
- **Handoff via Return**: Function returns Agent object to transfer control
- **Context Variables**: State passed through all agents and functions
- **Automatic Schema Generation**: Python type hints → JSON schema
- **Triage Pattern**: Central dispatcher routes to specialists

**Agent Definition:**
```python
agent = Agent(
    name="FileManager",
    instructions="Handle file operations",
    functions=[open_file, move_file, delete_file]
)
```

**Adoptable Ideas:**
- [x] Simple two-primitive design (agents + handoffs)
- [x] Functions as tools with type hints
- [x] Context variables for state passing
- [x] Triage agent for intent routing

---

### 6. modelcontextprotocol/servers ✅
**Type:** MCP tool servers

**Essential Servers for Aria:**

| Server | Tools | Use Case |
|--------|-------|----------|
| **filesystem** | 14 | File read/write/search |
| **memory** | 9 | Knowledge graph persistence |
| **fetch** | 1 | Web content retrieval |
| **git** | 12 | Repository operations |
| **time** | 2 | Timezone conversions |

**Community Servers:**
- Slack, Discord, Gmail, Telegram
- GitHub, GitLab, Notion, Airtable
- PostgreSQL, MongoDB, Snowflake

**Configuration:**
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/aria"]
    }
  }
}
```

**Adoptable Ideas:**
- [ ] Integrate filesystem, memory, fetch servers immediately
- [ ] Add git server for developer users
- [ ] Use MCP for all external tool integration

---

### 7. browser-use/browser-use ✅
**Type:** AI browser automation (75k+ stars)

**Key Patterns:**
- **Registry-Based Actions**: `@tools.action()` decorator
- **Event Bus Coordination**: `bubus` for service coordination
- **Multi-Strategy Element Detection**: Hash → XPath → Accessibility → Attributes
- **Vision Integration**: Screenshots with configurable inclusion modes
- **Action Primitives**: click, input_text, scroll, extract_content, navigate

**Action Registry Pattern:**
```python
@registry.action("Click at screen coordinates")
async def click(x: int, y: int, button: str = "left"):
    return await perform_click(x, y, button)
```

**Adoptable Ideas:**
- [x] Decorator-based action registry
- [ ] Multi-strategy element finding (OCR, accessibility, template)
- [ ] Event-driven architecture with bus
- [ ] Configurable vision modes (always/auto/never)

---

### 8. pipecat-ai/pipecat ✅
**Type:** Voice agent framework

**Key Patterns:**
- **Pipeline Architecture**: Composable audio/video processing stages
- **Transport Abstraction**: WebRTC, WebSocket, Daily, Twilio
- **STT/TTS Integration**: Deepgram, ElevenLabs, OpenAI, Azure
- **Real-time Processing**: Frame-based streaming
- **Interruption Handling**: Barge-in detection and response cancellation

**Adoptable Ideas:**
- [ ] Pipeline-based audio processing
- [ ] Transport abstraction for flexibility
- [ ] Frame-based streaming model
- [ ] Proper interruption handling

---

### 9. Significant-Gravitas/AutoGPT ✅
**Type:** Autonomous AI agents

**Key Patterns:**
- **Self-Prompting Loop**: Agent generates own next prompts
- **Dual Memory**: Short-term (context) + Long-term (vector DB)
- **ReAct Pattern**: Think → Act → Observe → Repeat
- **Reflection/Criticism**: Self-evaluation and course correction
- **Plugin Architecture**: Extensible capabilities via Python plugins

**Autonomy Levels:**
1. Supervised: Confirm every action
2. Semi-Autonomous: Routine tasks auto, sensitive tasks confirm
3. Autonomous: Full execution with reporting

**Adoptable Ideas:**
- [ ] Self-prompting for multi-step tasks
- [ ] Reflection loops for error recovery
- [ ] Configurable autonomy levels
- [ ] Plugin system for extensibility

---

### 10. geekan/MetaGPT ✅
**Type:** Multi-agent with software company roles

**Key Patterns:**
- **Role-Based Agents**: PM, Architect, Engineer, QA with defined responsibilities
- **SOP Encoding**: Standard procedures encoded in prompts
- **Watch-Based Subscription**: Roles subscribe to upstream message types
- **Structured Outputs**: Documents/schemas instead of natural language
- **React Modes**: REACT, BY_ORDER, PLAN_AND_ACT

**Role Definition:**
```python
class Role:
    name: str
    profile: str
    goal: str
    actions: List[Action]  # What it can do
    watch: List[Action]    # What triggers it
```

**Proposed Aria Roles:**
| Role | Trigger | Responsibility |
|------|---------|----------------|
| Coordinator | Always | Intent routing |
| AppController | "open/close/switch" | App lifecycle |
| FileManager | "find/move/create" | File operations |
| Researcher | "search/lookup" | Web research |
| Automator | "automate/repeat" | Workflow recording |

**Adoptable Ideas:**
- [ ] Role-based agent specialization
- [ ] Watch-based message routing
- [ ] SOP templates for common tasks
- [ ] Structured output protocol

---

### 11. OpenAI Realtime API ✅
**Type:** Real-time voice AI

---

### 12. obra/superpowers ✅
**Type:** Claude Code skills library

**Key Patterns:**
- **Skill Files with YAML Frontmatter**: Self-contained markdown files with metadata
- **Hook System**: Lifecycle hooks (session start, resume, clear) inject context
- **Skill Shadowing**: Personal skills override system skills, explicit namespacing
- **Seven-Stage Workflow**: Brainstorm → Plan → Execute → TDD → Review → Verify → Complete
- **Subagent Orchestration**: Dispatch parallel agents for complex tasks
- **Context Injection**: Hook output marked "EXTREMELY_IMPORTANT" for priority

**14 Skills Included:**
1. Test-Driven Development (RED-GREEN-REFACTOR)
2. Systematic Debugging (4-phase root cause analysis)
3. Brainstorming (Socratic design refinement)
4. Writing Plans (detailed specifications)
5. Executing Plans (batch with checkpoints)
6. Code Review (request/receive patterns)
7. Dispatching Parallel Agents
8. Subagent-Driven Development
9. Git Worktrees (parallel branches)
10. Finishing Development Branch
11. Verification Before Completion
12. Writing New Skills
13. Using Superpowers (intro)

**Skill File Format:**
```markdown
---
name: skill-name
description: What the skill does
triggers: ["keyword1", "keyword2"]
---
# Skill Content
Instructions here...
```

**Hook Configuration:**
```json
{
  "hooks": {
    "SessionStart": [{
      "matcher": "startup|resume",
      "hooks": [{"type": "command", "command": "inject-context.sh"}]
    }]
  }
}
```

**Adoptable Ideas:**
- [x] YAML frontmatter skill format (perfect for Aria!)
- [x] Hook system for session lifecycle
- [x] Skill shadowing for user customization
- [x] Context injection on session start
- [ ] Subagent dispatching for parallel work
- [ ] Verification patterns for safety-critical actions

---

### 11. OpenAI Realtime API ✅ (continued)
**Type:** Real-time voice AI

**Key Patterns:**
- **Native Speech-to-Speech**: No STT→LLM→TTS pipeline
- **WebSocket Streaming**: Bidirectional audio/text
- **Server VAD**: Offload speech detection to OpenAI
- **Interruption Handling**: Track playback, truncate context
- **Function Calling**: Tools integrated into voice flow

**Latency Targets:**
- Voice-to-voice: ~800ms ideal
- OpenAI TTFB: ~500ms
- Remaining budget: 300ms for audio processing

**Session Configuration:**
```json
{
  "turn_detection": {
    "type": "server_vad",
    "threshold": 0.5,
    "silence_duration_ms": 600
  },
  "tools": [
    {"type": "function", "name": "click", ...},
    {"type": "function", "name": "type_text", ...}
  ]
}
```

**Adoptable Ideas:**
- [ ] Upgrade to OpenAI Realtime API for voice
- [ ] Server-side VAD for better detection
- [ ] Proper interruption handling with context truncation
- [ ] Opus audio format for lower latency

---

## Synthesis: Recommended Aria Architecture

### Skills System Design

```python
# aria/skills/base.py
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class Skill:
    name: str
    description: str
    handler: Callable
    requires_confirmation: bool = False
    requires_screen: bool = False
    category: str = "general"

class SkillRegistry:
    _skills: dict[str, Skill] = {}

    @classmethod
    def skill(cls, name: str, **kwargs):
        """Decorator to register a skill."""
        def decorator(func):
            cls._skills[name] = Skill(
                name=name,
                handler=func,
                **kwargs
            )
            return func
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Skill]:
        return cls._skills.get(name)

    @classmethod
    def all(cls) -> list[Skill]:
        return list(cls._skills.values())
```

### Specialist Agents

```
┌─────────────────────────────────────────────────────────────┐
│                     ARIA COORDINATOR                         │
│                  (Intent Classification)                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  FileAgent    │ │  BrowserAgent │ │  SystemAgent  │
│  - find       │ │  - search     │ │  - open_app   │
│  - move       │ │  - navigate   │ │  - screenshot │
│  - create     │ │  - extract    │ │  - click      │
│  - delete     │ │  - fill_form  │ │  - type       │
└───────────────┘ └───────────────┘ └───────────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
                    MCP Servers
              (filesystem, fetch, memory)
```

### Workflow Graph

```
[Voice Input]
    │
    ▼
[Quick Classify] ──yes──▶ [Simple Action] ──▶ [Execute] ──▶ [Done]
    │ no
    ▼
[Intent Engine] ──▶ [Select Agent] ──▶ [Plan Steps]
    │                                      │
    ▼                                      ▼
[Clarify?] ──yes──▶ [Ask User]        [Execute Step]
    │ no                                   │
    ▼                                      ▼
[Execute Plan] ◀────────────────── [Verify Result]
    │                                      │
    ▼                              (loop if needed)
[Learn Outcome]
```

---

## Implementation Roadmap

### Phase 1: Skills Registry (Week 1)
- [ ] Create `aria/skills/` module
- [ ] Implement decorator-based registration
- [ ] Migrate existing actions to skills
- [ ] Add skill discovery for LLM

### Phase 2: MCP Integration (Week 1-2)
- [ ] Add MCP client to Aria
- [ ] Configure filesystem server
- [ ] Configure memory server (replace ChromaDB?)
- [ ] Configure fetch server

### Phase 3: Specialist Agents (Week 2-3)
- [ ] Create FileAgent with file skills
- [ ] Create BrowserAgent with web skills
- [ ] Create SystemAgent with desktop skills
- [ ] Implement Swarm-style handoffs

### Phase 4: Workflow Graphs (Week 3-4)
- [ ] Add LangGraph dependency
- [ ] Implement checkpointing
- [ ] Add human-in-the-loop nodes
- [ ] Create workflow templates

### Phase 5: Realtime Voice (Week 4-5)
- [ ] Integrate OpenAI Realtime API
- [ ] Implement server VAD
- [ ] Add interruption handling
- [ ] Optimize for <1s latency

---

## Capability Matrix

### 🖥️ Desktop Control
| Capability | Source | Priority | Complexity |
|------------|--------|----------|------------|
| Click/type/scroll | Browser-Use, OpenAdapt | HIGH | Low |
| App launching | Swarm, Eigent | HIGH | Low |
| Window management | OpenAdapt | MEDIUM | Medium |
| Clipboard operations | - | MEDIUM | Low |
| Recording workflows | OpenAdapt | LOW | High |

### 🌐 Web & Browser
| Capability | Source | Priority | Complexity |
|------------|--------|----------|------------|
| URL navigation | Browser-Use | HIGH | Low |
| Web search | MCP fetch, Browser-Use | HIGH | Low |
| Form filling | Browser-Use | MEDIUM | Medium |
| Content extraction | Browser-Use, MCP | MEDIUM | Medium |
| Multi-tab management | Browser-Use | LOW | Medium |

### 🗣️ Voice & Communication
| Capability | Source | Priority | Complexity |
|------------|--------|----------|------------|
| Real-time speech | OpenAI Realtime | HIGH | Medium |
| Interruption handling | Realtime, Pipecat | HIGH | Medium |
| Voice activity detection | Realtime | HIGH | Low |
| Multi-language support | - | LOW | High |

### 🤖 Agent Intelligence
| Capability | Source | Priority | Complexity |
|------------|--------|----------|------------|
| Intent understanding | Already built | HIGH | Done |
| Task planning | Already built | HIGH | Done |
| Learning from outcomes | Already built | HIGH | Done |
| Multi-agent handoffs | Swarm, AutoGen | MEDIUM | Medium |
| Self-prompting autonomy | AutoGPT | LOW | High |

### 🔧 Tools & Integrations
| Capability | Source | Priority | Complexity |
|------------|--------|----------|------------|
| File operations | MCP filesystem | HIGH | Low |
| Memory persistence | MCP memory | HIGH | Low |
| Git operations | MCP git | MEDIUM | Low |
| Calendar/email | MCP community | LOW | Medium |
| Slack/Discord | MCP community | LOW | Medium |

---

---

### 13. google-ai-edge/mediapipe ✅
**Type:** On-device ML for vision/audio

**Key Patterns:**
- **100% Local Processing**: No API calls, no data leaves device
- **Detection-Tracking Separation**: Heavy detection once, lightweight tracking per frame
- **30 FPS on CPU**: Real-time performance without GPU
- **Pre-trained Models**: No ML training required

**Solutions Available:**
| Solution | Landmarks | Use Case for Aria |
|----------|-----------|-------------------|
| Face Detection | 6 keypoints | User presence detection |
| Face Mesh | 468 3D | Attention/gaze tracking |
| Hand Tracking | 21 per hand | Gesture control |
| Gesture Recognition | 7 gestures | Hands-free commands |
| Pose Estimation | 33 body | Full body awareness |
| Holistic | 543 total | Combined multimodal |

**Built-in Gestures:** `Closed_Fist`, `Open_Palm`, `Pointing_Up`, `Thumb_Down`, `Thumb_Up`, `Victory`, `ILoveYou`

**Adoptable Ideas:**
- [ ] Gesture control alongside voice ("thumbs up" = confirm)
- [ ] Face presence detection (wake on presence, sleep on away)
- [ ] Attention tracking (pause when user looks away)
- [ ] Visual pointing ("Aria, what's that?" + point at screen)

---

### 14. bytedance/UI-TARS-desktop ✅
**Type:** AI desktop UI automation

**Key Patterns:**
- **End-to-End VLM**: Single model for perception + reasoning + action
- **Unified Operator Interface**: Clean abstraction for cross-platform control
- **Thought-Action Output**: Explicit reasoning before every action
- **Normalized Coordinates**: 0-1000 scale for resolution independence

**Action Primitives:**
```
click, type, scroll, drag, hotkey, wait, finished, call_user
left_double, right_single (desktop-specific)
```

**Thought-Action Format:**
```
Thought: I need to click the Submit button in the bottom right.
Action: click(start_box='(850,720)')
```

**Adoptable Ideas:**
- [x] Add explicit thought/reasoning to Aria's output
- [ ] Unified operator abstraction for control layer
- [ ] Normalized coordinate system (resolution-independent)
- [ ] "call_user" action for requesting help

---

### 15. NevaMind-AI/memU ✅
**Type:** Hierarchical memory for AI agents

**Key Patterns:**
- **Three-Layer Hierarchy**: Categories → Items → Resources
- **Full Traceability**: Track from raw data to summaries and back
- **Dual Retrieval**: RAG (fast) vs LLM (deep) per query
- **Self-Evolving Structure**: Categories adapt based on usage

**Memory Types:** `profile`, `event`, `knowledge`, `behavior`, `skill`

**Architecture:**
```
Category Layer  → Aggregated summaries (preferences.md, work_life.md)
Item Layer      → Discrete memory units with embeddings
Resource Layer  → Raw multimodal data (conversations, docs, images)
```

**Adoptable Ideas:**
- [ ] Hierarchical organization (Categories → Items → Raw)
- [ ] Dual retrieval strategy (fast RAG + deep LLM)
- [ ] Memory type schema for voice context
- [ ] Sufficiency checking (stop early when answer found)

---

### 16. memvid/memvid ✅
**Type:** Single-file video-based memory

**Key Patterns:**
- **Single-File Storage**: Entire knowledge base in one `.mv2` file
- **Smart Frames**: Immutable units with content + timestamp + checksum
- **Five Index Types**: BM25, Vector, SimHash, Time, Logic Mesh
- **Sub-5ms Queries**: On 1M+ chunks

**Index Types:**
| Index | Purpose |
|-------|---------|
| BM25 (Tantivy) | Exact keyword matching |
| HNSW Vectors | Semantic similarity |
| SimHash | O(1) deduplication |
| Time B-tree | Temporal queries |
| Logic Mesh | Entity relationships |

**Adoptable Ideas for Screen Memory:**
- [ ] Single-file memory format (`.aria`)
- [ ] Smart frames for screenshots (timestamp + OCR + CLIP + metadata)
- [ ] SimHash for duplicate detection (skip similar screenshots)
- [ ] Multi-index fusion (BM25 + semantic + temporal)
- [ ] "What was I doing at 3pm?" temporal queries

---

### 17. ChromeDevTools/chrome-devtools-mcp ✅
**Type:** MCP server for Chrome automation

**Key Patterns:**
- **26 Tools**: Input, navigation, debugging, network, performance
- **Full DOM Access**: Read/write DOM, query selectors, computed styles
- **Network Inspection**: Monitor all HTTP requests/responses
- **Auto-Connect**: Chrome 144+ automatic discovery

**Tool Categories:**
| Category | Tools |
|----------|-------|
| Input | click, drag, fill, fill_form, hover, press_key, upload_file |
| Navigation | navigate_page, new_page, close_page, list_pages, wait_for |
| Debugging | evaluate_script, take_screenshot, take_snapshot, console |
| Network | list_network_requests, get_network_request |
| Performance | start_trace, stop_trace, analyze_insight |

**Adoptable Ideas:**
- [ ] Semantic element selection (by selector, not just coordinates)
- [ ] Reliable wait conditions (wait for element, not time delay)
- [ ] Console/network monitoring for debugging help
- [ ] Fill forms by field name, not coordinates

---

## Updated Synthesis: Recommended Aria Architecture

### NEW: Multimodal Input Layer

```
┌─────────────────────────────────────────────────────────────┐
│                    ARIA INPUT LAYER                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   VOICE     │  │   VISION    │  │   GESTURE          │  │
│  │   (Whisper) │  │   (Screen)  │  │   (MediaPipe)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│        │                │                    │               │
│        │     ┌──────────┴──────────┐        │               │
│        │     │  UI Understanding   │        │               │
│        │     │  (UI-TARS pattern)  │        │               │
│        │     └──────────┬──────────┘        │               │
│        │                │                    │               │
│        └────────────────┼────────────────────┘               │
│                         │                                    │
│              ┌──────────▼──────────┐                        │
│              │  MULTIMODAL FUSION  │                        │
│              │  Voice + Screen +   │                        │
│              │  Gesture = Intent   │                        │
│              └─────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### NEW: Hierarchical Memory System

```
┌─────────────────────────────────────────────────────────────┐
│                    ARIA MEMORY (.aria file)                  │
├─────────────────────────────────────────────────────────────┤
│  CATEGORY LAYER                                              │
│  ├── preferences.md (user preferences summary)               │
│  ├── commands.md (learned voice commands)                    │
│  └── context.md (current projects/tasks)                     │
├─────────────────────────────────────────────────────────────┤
│  ITEM LAYER (with embeddings)                                │
│  ├── Individual facts, preferences, commands                 │
│  └── Searchable via RAG or LLM                              │
├─────────────────────────────────────────────────────────────┤
│  RESOURCE LAYER                                              │
│  ├── Voice transcripts                                       │
│  ├── Screen captures (Smart Frames)                          │
│  └── Interaction logs                                        │
├─────────────────────────────────────────────────────────────┤
│  INDICES                                                     │
│  ├── Vector (semantic search)                                │
│  ├── BM25 (keyword search)                                   │
│  ├── Time (temporal queries)                                 │
│  └── SimHash (deduplication)                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Updated Implementation Roadmap

### Phase 1: Skills Registry (Week 1)
- [ ] Markdown skills with YAML frontmatter (Superpowers pattern)
- [ ] Hook system for session lifecycle
- [ ] Skill shadowing for user customization

### Phase 2: Enhanced Memory (Week 1-2)
- [ ] Hierarchical memory (memU pattern)
- [ ] Single-file storage (memvid pattern)
- [ ] Multi-index search (BM25 + vector + time)

### Phase 3: MCP Integration (Week 2)
- [ ] filesystem, memory, fetch servers
- [ ] Chrome DevTools MCP for browser control
- [ ] Semantic element selection (not just coordinates)

### Phase 4: Multimodal Input (Week 3)
- [ ] MediaPipe gesture recognition
- [ ] Face presence detection
- [ ] Gesture + voice fusion ("thumbs up" = confirm)

### Phase 5: UI Understanding (Week 3-4)
- [ ] Thought-action output format (UI-TARS pattern)
- [ ] Normalized coordinate system
- [ ] Unified operator abstraction

### Phase 6: Realtime Voice (Week 4-5)
- [ ] OpenAI Realtime API integration
- [ ] Server-side VAD
- [ ] Sub-second latency

---

## Next Steps

1. **Review this document** and prioritize capabilities
2. **Add more repos** to analyze if needed
3. **Start Phase 1** - Skills Registry implementation
4. **Design skill discovery** - How will Aria find new skills on GitHub?

