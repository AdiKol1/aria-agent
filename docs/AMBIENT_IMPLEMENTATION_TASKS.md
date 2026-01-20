# Ambient Intelligence - Implementation Tasks

> **Task Breakdown for Parallel Subagent Execution**
> Reference: [AMBIENT_INTELLIGENCE_PRD.md](./AMBIENT_INTELLIGENCE_PRD.md)

---

## Overview

This document breaks down the Ambient Intelligence implementation into discrete, parallelizable tasks that can be assigned to multiple subagents.

**Total Estimated Effort:** 6-8 weeks
**Parallelization Factor:** Up to 5 concurrent agents

---

## Task Dependencies Graph

```
                    ┌─────────────────┐
                    │  PHASE 1        │
                    │  Data Models    │
                    │  (Foundation)   │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │ PHASE 2A      │ │ PHASE 2B      │ │ PHASE 2C      │
    │ World Manager │ │ Watcher Base  │ │ Storage       │
    └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
            │                 │                 │
            │    ┌────────────┴─────────┐      │
            │    │                      │      │
            ▼    ▼                      ▼      │
    ┌───────────────┐           ┌───────────────┐
    │ PHASE 3A      │           │ PHASE 3B      │
    │ Cortex        │           │ Watchers      │
    │ (Synthesis)   │           │ (Specific)    │
    └───────┬───────┘           └───────┬───────┘
            │                           │
            └─────────────┬─────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ PHASE 4       │
                  │ Actors +      │
                  │ Delivery      │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ PHASE 5       │
                  │ Integration   │
                  └───────────────┘
```

---

## Phase 1: Foundation (Data Models)

**Duration:** 2-3 days
**Parallelization:** 1 agent (blocking for others)
**Priority:** CRITICAL - Must complete first

### Task 1.1: Core Data Models

**File:** `aria/ambient/models.py`

**Description:** Define all data classes for the ambient system.

**Deliverables:**
```python
# Classes to implement:
@dataclass class World
@dataclass class Goal
@dataclass class Entity
@dataclass class Signal
@dataclass class Insight
@dataclass class PreparedAction
@dataclass class WorldMatch
@dataclass class Connection
@dataclass class QuickAction
```

**Acceptance Criteria:**
- [ ] All dataclasses defined with full type hints
- [ ] Serialization methods (to_dict, from_dict)
- [ ] Validation methods where needed
- [ ] Unit tests for all models

**Reference:** PRD Section 5 (Data Models)

---

### Task 1.2: Enums and Constants

**File:** `aria/ambient/constants.py`

**Description:** Define enums and constants.

**Deliverables:**
```python
class SignalType(Enum)      # news, mention, price_change, etc.
class InsightPriority(Enum) # critical, high, medium, low
class ActionType(Enum)      # draft, respond, research, etc.
class DeliveryChannel(Enum) # push, voice, desktop, digest
class EntityType(Enum)      # person, company, topic, etc.
class RelationshipType(Enum)# competitor, client, partner, etc.

# Constants
PRIORITY_THRESHOLDS = {...}
CHECK_INTERVALS = {...}
```

**Acceptance Criteria:**
- [ ] All enums defined
- [ ] Constants documented
- [ ] Used consistently in models

---

### Task 1.3: Package Structure

**File:** `aria/ambient/__init__.py` and directory setup

**Description:** Create the ambient package structure.

**Deliverables:**
```
aria/ambient/
├── __init__.py
├── models.py
├── constants.py
├── world_manager.py
├── storage.py
├── relevance.py
├── cortex/
│   ├── __init__.py
│   ├── synthesis.py
│   ├── priority.py
│   └── consciousness.py
├── watchers/
│   ├── __init__.py
│   ├── base.py
│   └── [specific watchers]
├── actors/
│   ├── __init__.py
│   ├── base.py
│   └── [specific actors]
└── delivery/
    ├── __init__.py
    └── engine.py
```

**Acceptance Criteria:**
- [ ] All directories created
- [ ] __init__.py files with proper exports
- [ ] Package importable

---

## Phase 2A: World Manager

**Duration:** 3-4 days
**Parallelization:** Can start after Phase 1
**Dependencies:** Phase 1

### Task 2A.1: World CRUD Operations

**File:** `aria/ambient/world_manager.py`

**Description:** Implement World create, read, update, delete operations.

**Deliverables:**
```python
class WorldManager:
    def __init__(self, storage_path: Path)

    # CRUD
    def create_world(self, name: str, description: str) -> World
    def get_world(self, world_id: str) -> Optional[World]
    def update_world(self, world_id: str, updates: dict) -> bool
    def delete_world(self, world_id: str) -> bool
    def list_worlds(self) -> List[World]

    # Goals
    def add_goal(self, world_id: str, goal: Goal) -> bool
    def update_goal(self, world_id: str, goal_id: str, updates: dict) -> bool
    def remove_goal(self, world_id: str, goal_id: str) -> bool

    # Entities
    def add_entity(self, world_id: str, entity: Entity) -> bool
    def update_entity(self, world_id: str, entity_id: str, updates: dict) -> bool
    def remove_entity(self, world_id: str, entity_id: str) -> bool
```

**Acceptance Criteria:**
- [ ] All CRUD operations work
- [ ] Data persisted to storage
- [ ] Error handling for invalid IDs
- [ ] Unit tests with >80% coverage

---

### Task 2A.2: World Context Detection

**File:** `aria/ambient/world_manager.py` (continued)

**Description:** Implement logic to determine which world(s) are currently active.

**Deliverables:**
```python
class WorldManager:
    # Context detection
    def get_active_worlds(self) -> List[World]
    def get_world_for_context(self, context: dict) -> Optional[World]
    def is_world_active(self, world_id: str) -> bool

    # Based on:
    # - Time of day / day of week (schedule)
    # - Current screen context
    # - Recent activity
    # - Keywords in recent input
```

**Acceptance Criteria:**
- [ ] Time-based activation works
- [ ] Context keywords trigger correct world
- [ ] Multiple active worlds supported
- [ ] Tests for various scenarios

---

### Task 2A.3: World Learning

**File:** `aria/ambient/world_learning.py`

**Description:** Implement learning logic to infer and suggest worlds from user behavior.

**Deliverables:**
```python
class WorldLearner:
    def observe_activity(self, activity: UserActivity) -> None
    def infer_worlds(self) -> List[WorldSuggestion]
    def suggest_entities(self, world_id: str) -> List[EntitySuggestion]
    def suggest_goals(self, world_id: str) -> List[GoalSuggestion]
    def confirm_suggestion(self, suggestion_id: str) -> bool
```

**Acceptance Criteria:**
- [ ] Activity patterns tracked
- [ ] World suggestions generated
- [ ] User can confirm/reject suggestions
- [ ] Suggestions improve over time

---

## Phase 2B: Watcher Base

**Duration:** 2-3 days
**Parallelization:** Can start after Phase 1
**Dependencies:** Phase 1

### Task 2B.1: Watcher Abstract Base Class

**File:** `aria/ambient/watchers/base.py`

**Description:** Define the base class for all watchers.

**Deliverables:**
```python
class Watcher(ABC):
    name: str
    description: str
    check_interval: int  # seconds
    enabled: bool

    @abstractmethod
    async def observe(self) -> List[Signal]

    def configure(self, settings: dict) -> None
    def enable(self) -> None
    def disable(self) -> None
    def get_status(self) -> dict
```

**Acceptance Criteria:**
- [ ] ABC properly defined
- [ ] All abstract methods declared
- [ ] Configuration system works
- [ ] Status reporting works

---

### Task 2B.2: Watcher Scheduler

**File:** `aria/ambient/watchers/scheduler.py`

**Description:** Background scheduler to run watchers at their intervals.

**Deliverables:**
```python
class WatcherScheduler:
    def __init__(self)

    def register_watcher(self, watcher: Watcher) -> None
    def unregister_watcher(self, name: str) -> None

    async def start(self) -> None
    async def stop(self) -> None

    def get_all_signals(self) -> List[Signal]
    def get_status(self) -> dict
```

**Acceptance Criteria:**
- [ ] Watchers run at correct intervals
- [ ] Signals collected and queued
- [ ] Graceful start/stop
- [ ] Error handling per watcher
- [ ] Status reporting

---

## Phase 2C: Storage

**Duration:** 2-3 days
**Parallelization:** Can start after Phase 1
**Dependencies:** Phase 1

### Task 2C.1: World Storage

**File:** `aria/ambient/storage.py`

**Description:** YAML-based storage for worlds.

**Deliverables:**
```python
class WorldStorage:
    def __init__(self, storage_path: Path)

    def save_world(self, world: World) -> bool
    def load_world(self, world_id: str) -> Optional[World]
    def delete_world(self, world_id: str) -> bool
    def list_world_ids(self) -> List[str]
```

**Acceptance Criteria:**
- [ ] Worlds saved as YAML files
- [ ] Atomic writes (temp file + rename)
- [ ] Backup on update
- [ ] Migration support for schema changes

---

### Task 2C.2: Signal Cache

**File:** `aria/ambient/storage.py` (continued)

**Description:** Temporary storage for signals before processing.

**Deliverables:**
```python
class SignalCache:
    def __init__(self, max_age: int = 3600)  # 1 hour default

    def add_signal(self, signal: Signal) -> None
    def get_signals(self, since: datetime = None) -> List[Signal]
    def clear_old(self) -> int  # returns count cleared
```

**Acceptance Criteria:**
- [ ] Signals cached in memory
- [ ] Automatic expiration
- [ ] Thread-safe operations

---

### Task 2C.3: Insight History

**File:** `aria/ambient/storage.py` (continued)

**Description:** Persistent storage for insights and their outcomes.

**Deliverables:**
```python
class InsightHistory:
    def __init__(self, storage_path: Path)

    def record_insight(self, insight: Insight) -> None
    def record_outcome(self, insight_id: str, outcome: str) -> None
    def get_recent(self, days: int = 7) -> List[Insight]
    def get_by_world(self, world_id: str) -> List[Insight]
```

**Acceptance Criteria:**
- [ ] Insights persisted to JSON
- [ ] Outcome tracking works
- [ ] Query by time and world
- [ ] Cleanup of old data

---

## Phase 3A: Cortex (Synthesis)

**Duration:** 4-5 days
**Parallelization:** Can start after Phase 2A, 2B
**Dependencies:** Phase 2A, 2B

### Task 3A.1: Relevance Scoring

**File:** `aria/ambient/relevance.py`

**Description:** Calculate relevance of signals to worlds.

**Deliverables:**
```python
class RelevanceScorer:
    def score_signal(self, signal: Signal, world: World) -> WorldMatch
    def score_entity_match(self, signal: Signal, entity: Entity) -> float
    def score_keyword_match(self, signal: Signal, keywords: List[str]) -> float
    def score_goal_relevance(self, signal: Signal, goal: Goal) -> float
```

**Acceptance Criteria:**
- [ ] Entity matching works
- [ ] Keyword matching works
- [ ] Goal relevance calculated
- [ ] Scores normalized 0-1
- [ ] Unit tests for edge cases

---

### Task 3A.2: Priority Calculation

**File:** `aria/ambient/cortex/priority.py`

**Description:** Calculate priority of insights.

**Deliverables:**
```python
class PriorityCalculator:
    def calculate(self, signal: Signal, world: World) -> float
    def calculate_urgency(self, signal: Signal) -> float
    def calculate_actionability(self, signal: Signal) -> float
    def calculate_novelty(self, signal: Signal, history: List) -> float
```

**Acceptance Criteria:**
- [ ] Priority formula implemented
- [ ] All factors weighted correctly
- [ ] Novelty detection works
- [ ] Tests for priority edge cases

---

### Task 3A.3: Insight Generation

**File:** `aria/ambient/cortex/synthesis.py`

**Description:** Generate human-readable insights from signals.

**Deliverables:**
```python
class InsightGenerator:
    def __init__(self, llm_client)

    async def generate_insight(
        self,
        signal: Signal,
        world: World,
        context: dict
    ) -> Insight

    async def suggest_action(self, insight: Insight) -> str
```

**Acceptance Criteria:**
- [ ] Insights are clear and actionable
- [ ] LLM used for summarization
- [ ] Action suggestions relevant
- [ ] Fallback for LLM failures

---

### Task 3A.4: Cross-World Connections

**File:** `aria/ambient/cortex/connections.py`

**Description:** Detect opportunities across worlds.

**Deliverables:**
```python
class ConnectionDetector:
    def find_connections(
        self,
        signals: List[Signal],
        worlds: List[World]
    ) -> List[Connection]

    def score_connection(self, connection: Connection) -> float
```

**Acceptance Criteria:**
- [ ] Cross-world patterns detected
- [ ] Connections scored by value
- [ ] Duplicates filtered
- [ ] Tests with multi-world scenarios

---

### Task 3A.5: Consciousness Stream

**File:** `aria/ambient/cortex/consciousness.py`

**Description:** Maintain Aria's running internal state.

**Deliverables:**
```python
class ConsciousnessStream:
    current_focus: List[str]
    active_insights: List[Insight]
    pending_preparations: List[str]

    def update(self, new_insights: List[Insight]) -> None
    def get_briefing(self) -> str
    def get_focus() -> List[str]
    def log_thought(self, thought: str) -> None
```

**Acceptance Criteria:**
- [ ] State maintained across cycles
- [ ] Briefing generation works
- [ ] Thought logging for debugging
- [ ] Thread-safe updates

---

## Phase 3B: Specific Watchers

**Duration:** 4-5 days
**Parallelization:** Can run parallel with 3A
**Dependencies:** Phase 2B

### Task 3B.1: News Watcher

**File:** `aria/ambient/watchers/news.py`

**Description:** Monitor RSS feeds and news APIs.

**Deliverables:**
```python
class NewsWatcher(Watcher):
    name = "news"

    def __init__(self, feeds: List[str], api_keys: dict = None)
    async def observe(self) -> List[Signal]
    def add_feed(self, url: str) -> None
    def remove_feed(self, url: str) -> None
```

**Acceptance Criteria:**
- [ ] RSS parsing works
- [ ] Multiple feeds supported
- [ ] NewsAPI integration (optional)
- [ ] Deduplication of articles
- [ ] Error handling for feed failures

---

### Task 3B.2: Calendar Watcher

**File:** `aria/ambient/watchers/calendar.py`

**Description:** Monitor calendar for upcoming events.

**Deliverables:**
```python
class CalendarWatcher(Watcher):
    name = "calendar"

    def __init__(self, calendar_source: str)  # "google", "apple", etc.
    async def observe(self) -> List[Signal]
    def get_upcoming(self, days: int = 7) -> List[Signal]
```

**Acceptance Criteria:**
- [ ] Google Calendar integration
- [ ] Apple Calendar integration (optional)
- [ ] Event signals generated
- [ ] Deadline detection

---

### Task 3B.3: Screen Context Watcher

**File:** `aria/ambient/watchers/screen.py`

**Description:** Monitor current screen context for world detection.

**Deliverables:**
```python
class ScreenContextWatcher(Watcher):
    name = "screen"

    async def observe(self) -> List[Signal]
    def get_active_app(self) -> str
    def get_active_url(self) -> Optional[str]  # If browser
    def is_user_present(self) -> bool
    def is_user_focused(self) -> bool  # Not in meeting, etc.
```

**Acceptance Criteria:**
- [ ] Active app detection
- [ ] Browser URL detection
- [ ] Presence detection
- [ ] Focus state detection
- [ ] Privacy: No content capture

---

### Task 3B.4: Web/Social Watcher (Phase 2 - Optional for MVP)

**File:** `aria/ambient/watchers/social.py`

**Description:** Monitor social media (within API limits).

**Deliverables:**
```python
class SocialWatcher(Watcher):
    name = "social"

    def __init__(self, accounts: dict, api_keys: dict)
    async def observe(self) -> List[Signal]
    def track_account(self, platform: str, handle: str) -> None
```

**Note:** May require creative solutions due to API restrictions.

**Acceptance Criteria:**
- [ ] Twitter/X monitoring (if API available)
- [ ] LinkedIn monitoring (if API available)
- [ ] Graceful degradation if APIs unavailable
- [ ] Rate limiting

---

## Phase 4: Actors & Delivery

**Duration:** 4-5 days
**Parallelization:** Can start after Phase 3A
**Dependencies:** Phase 3A

### Task 4.1: Actor Base Class

**File:** `aria/ambient/actors/base.py`

**Description:** Base class for action preparers.

**Deliverables:**
```python
class Actor(ABC):
    name: str

    @abstractmethod
    async def prepare(self, insight: Insight) -> PreparedAction

    def can_handle(self, insight: Insight) -> bool
```

**Acceptance Criteria:**
- [ ] ABC properly defined
- [ ] Can_handle routing logic
- [ ] PreparedAction format standard

---

### Task 4.2: Content Drafter

**File:** `aria/ambient/actors/content.py`

**Description:** Draft social media posts, threads, responses.

**Deliverables:**
```python
class ContentDrafter(Actor):
    name = "content"

    async def prepare(self, insight: Insight) -> PreparedAction
    async def draft_post(self, topic: str, platform: str) -> str
    async def draft_thread(self, topic: str) -> List[str]
    async def draft_response(self, original: str, tone: str) -> str
```

**Acceptance Criteria:**
- [ ] Post drafts match platform constraints
- [ ] Multiple options generated
- [ ] Tone/voice customizable
- [ ] LLM integration works

---

### Task 4.3: Alert Composer

**File:** `aria/ambient/actors/alert.py`

**Description:** Compose alert messages for various channels.

**Deliverables:**
```python
class AlertComposer(Actor):
    name = "alert"

    async def prepare(self, insight: Insight) -> PreparedAction
    def compose_push(self, insight: Insight) -> str
    def compose_voice(self, insight: Insight) -> str
    def compose_digest_item(self, insight: Insight) -> str
```

**Acceptance Criteria:**
- [ ] Push notifications concise
- [ ] Voice briefings conversational
- [ ] Digest items scannable
- [ ] Character limits respected

---

### Task 4.4: Delivery Engine

**File:** `aria/ambient/delivery/engine.py`

**Description:** Deliver insights at the right time via right channel.

**Deliverables:**
```python
class DeliveryEngine:
    def __init__(self, channels: List[Channel])

    def queue(self, action: PreparedAction) -> None
    async def process_queue(self) -> None
    def select_channel(self, action: PreparedAction, context: dict) -> Channel
    def should_deliver_now(self, action: PreparedAction) -> bool
```

**Acceptance Criteria:**
- [ ] Queue management works
- [ ] Channel selection logic
- [ ] Timing logic (not during focus)
- [ ] Delivery confirmation

---

### Task 4.5: Digest Compiler

**File:** `aria/ambient/delivery/digest.py`

**Description:** Compile periodic digests from queued insights.

**Deliverables:**
```python
class DigestCompiler:
    def compile_morning(self, insights: List[Insight]) -> str
    def compile_evening(self, insights: List[Insight]) -> str
    def compile_weekly(self, insights: List[Insight]) -> str
```

**Acceptance Criteria:**
- [ ] Digests well-organized
- [ ] Grouped by world
- [ ] Priority ordering
- [ ] Actionable format

---

## Phase 5: Integration

**Duration:** 3-4 days
**Parallelization:** Final phase, needs all above
**Dependencies:** All previous phases

### Task 5.1: Main Loop Integration

**File:** `aria/ambient/loop.py`

**Description:** The main ambient loop that ties everything together.

**Deliverables:**
```python
class AmbientLoop:
    def __init__(
        self,
        world_manager: WorldManager,
        watchers: List[Watcher],
        cortex: Cortex,
        actors: List[Actor],
        delivery: DeliveryEngine
    )

    async def start(self) -> None
    async def stop(self) -> None
    async def run_cycle(self) -> None
    def get_status(self) -> dict
```

**Acceptance Criteria:**
- [ ] All components wired together
- [ ] Cycle runs at correct interval
- [ ] Error handling doesn't crash loop
- [ ] Status reporting works

---

### Task 5.2: Aria Integration

**File:** `aria/main.py` (modifications)

**Description:** Integrate ambient system into main Aria.

**Deliverables:**
- [ ] Ambient loop starts with Aria
- [ ] Voice briefing triggers ambient data
- [ ] "What's going on?" queries consciousness stream
- [ ] New tools: `schedule_world`, `add_entity`, etc.

**Acceptance Criteria:**
- [ ] Ambient runs in background
- [ ] Voice activation works with ambient
- [ ] No performance degradation
- [ ] Graceful degradation if ambient fails

---

### Task 5.3: Onboarding Flow

**File:** `aria/ambient/onboarding.py`

**Description:** Guided setup for new users.

**Deliverables:**
```python
class AmbientOnboarding:
    async def start_onboarding(self) -> None
    async def discover_worlds(self) -> List[WorldSuggestion]
    async def setup_world(self, world_id: str) -> None
    async def complete_onboarding(self) -> None
```

**Acceptance Criteria:**
- [ ] Conversational setup flow
- [ ] World discovery works
- [ ] Goal/entity collection
- [ ] Can skip/resume

---

### Task 5.4: Testing & Documentation

**Files:** `tests/ambient/`, `docs/`

**Description:** Comprehensive tests and documentation.

**Deliverables:**
- [ ] Unit tests for all components
- [ ] Integration tests for full flow
- [ ] User documentation
- [ ] Developer documentation

**Acceptance Criteria:**
- [ ] >80% code coverage
- [ ] All edge cases tested
- [ ] Docs up to date
- [ ] Examples included

---

## Subagent Assignment Matrix

| Task | Complexity | Dependencies | Suggested Agent |
|------|------------|--------------|-----------------|
| 1.1 Data Models | Medium | None | Agent 1 |
| 1.2 Enums | Low | None | Agent 1 |
| 1.3 Package Structure | Low | None | Agent 1 |
| 2A.1 World CRUD | Medium | Phase 1 | Agent 2 |
| 2A.2 Context Detection | Medium | Phase 1 | Agent 2 |
| 2A.3 World Learning | High | Phase 1 | Agent 2 |
| 2B.1 Watcher Base | Low | Phase 1 | Agent 3 |
| 2B.2 Watcher Scheduler | Medium | Phase 1 | Agent 3 |
| 2C.1 World Storage | Low | Phase 1 | Agent 4 |
| 2C.2 Signal Cache | Low | Phase 1 | Agent 4 |
| 2C.3 Insight History | Low | Phase 1 | Agent 4 |
| 3A.1 Relevance Scoring | Medium | 2A, 2B | Agent 2 |
| 3A.2 Priority Calc | Medium | 2A | Agent 2 |
| 3A.3 Insight Generation | High | 2A, 2B | Agent 5 |
| 3A.4 Cross-World | High | 2A | Agent 5 |
| 3A.5 Consciousness | Medium | 3A.3 | Agent 5 |
| 3B.1 News Watcher | Medium | 2B | Agent 3 |
| 3B.2 Calendar Watcher | Medium | 2B | Agent 3 |
| 3B.3 Screen Watcher | Low | 2B | Agent 3 |
| 4.1 Actor Base | Low | 3A | Agent 4 |
| 4.2 Content Drafter | High | 4.1 | Agent 4 |
| 4.3 Alert Composer | Medium | 4.1 | Agent 4 |
| 4.4 Delivery Engine | Medium | 4.1 | Agent 4 |
| 4.5 Digest Compiler | Medium | 4.4 | Agent 4 |
| 5.1 Main Loop | High | All | Agent 1 |
| 5.2 Aria Integration | High | 5.1 | Agent 1 |
| 5.3 Onboarding | Medium | 5.2 | Agent 5 |
| 5.4 Testing | Medium | All | All |

---

## Execution Order

### Week 1
- **Agent 1:** Phase 1 (all tasks) → blocks everyone
- **Other agents:** Review PRD, prepare

### Week 2
- **Agent 2:** Phase 2A (World Manager)
- **Agent 3:** Phase 2B (Watcher Base)
- **Agent 4:** Phase 2C (Storage)

### Week 3
- **Agent 2:** Phase 3A.1, 3A.2 (Relevance, Priority)
- **Agent 3:** Phase 3B.1, 3B.2 (News, Calendar watchers)
- **Agent 4:** Phase 4.1 (Actor Base)
- **Agent 5:** Phase 3A.3 (Insight Generation)

### Week 4
- **Agent 2:** Continue 3A, start 3A.4
- **Agent 3:** Phase 3B.3, 3B.4 (Screen, Social watchers)
- **Agent 4:** Phase 4.2, 4.3 (Content, Alert)
- **Agent 5:** Phase 3A.4, 3A.5 (Connections, Consciousness)

### Week 5
- **Agent 4:** Phase 4.4, 4.5 (Delivery)
- **All:** Integration prep, testing

### Week 6
- **Agent 1:** Phase 5.1, 5.2 (Main Loop, Integration)
- **Agent 5:** Phase 5.3 (Onboarding)
- **All:** Testing, bug fixes

### Week 7-8
- **All:** Polish, documentation, user testing

---

## Success Criteria Summary

### MVP (Week 6)
- [ ] At least 2 worlds can be configured
- [ ] News and Calendar watchers working
- [ ] Basic insights generated
- [ ] Voice briefing includes ambient data
- [ ] Morning digest compiles

### Full Release (Week 8)
- [ ] All watchers operational
- [ ] Cross-world connections detected
- [ ] Content drafting works
- [ ] Delivery engine with timing logic
- [ ] Onboarding flow complete
- [ ] >80% test coverage

---

*End of Implementation Tasks*
