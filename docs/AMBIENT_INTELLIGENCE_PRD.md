# Aria 3.0: Ambient Intelligence PRD

> **Product Requirements Document**
> Version: 1.0
> Created: 2026-01-17
> Status: Draft

---

## Executive Summary

Aria evolves from a **reactive voice assistant** to **ambient intelligence** - an AI that thinks about your world even when you're not talking to it. Instead of waiting for commands, Aria continuously observes, synthesizes, and prepares actions across all domains of your life.

**The Core Shift:**
```
Aria 1.0: You ask → Aria responds
Aria 2.0: Aria learns from interactions → Better responses
Aria 3.0: Aria thinks continuously → Proactive intelligence
```

---

## Table of Contents

1. [Vision](#1-vision)
2. [The World System](#2-the-world-system)
3. [Architecture Overview](#3-architecture-overview)
4. [Core Components](#4-core-components)
5. [Data Models](#5-data-models)
6. [User Experience](#6-user-experience)
7. [Technical Requirements](#7-technical-requirements)
8. [Implementation Phases](#8-implementation-phases)
9. [Success Metrics](#9-success-metrics)
10. [Open Questions](#10-open-questions)

---

## 1. Vision

### 1.1 The Problem

Current AI assistants are **reactive command executors**:
- You ask a question → They answer
- You give a command → They execute
- They have no initiative, no ongoing awareness, no preparation

This is fundamentally different from how valuable human assistants work. A great executive assistant:
- Notices things before you ask
- Prepares materials before meetings
- Surfaces opportunities you'd miss
- Thinks about your goals when you're not in the room

### 1.2 The Solution: Ambient Intelligence

Aria becomes a **second mind running in parallel**:

```
┌─────────────────────────────────────────────────────────────────┐
│                     THE AMBIENT LOOP                            │
│                                                                 │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐               │
│   │ OBSERVE │ ───► │ THINK   │ ───► │ PREPARE │               │
│   │         │      │         │      │         │               │
│   │ Watch   │      │ What    │      │ Draft   │               │
│   │ signals │      │ matters?│      │ actions │               │
│   └─────────┘      └─────────┘      └─────────┘               │
│        ▲                                 │                     │
│        │                                 ▼                     │
│        │           ┌─────────┐                                │
│        └────────── │ DELIVER │ ◄──── (right time, right way)  │
│                    └─────────┘                                │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Differentiators

| Traditional Assistant | Ambient Intelligence |
|-----------------------|----------------------|
| Responds to commands | Initiates proactively |
| Single domain focus | Multi-world awareness |
| Forgets between sessions | Continuous memory |
| Generic responses | Deeply personalized |
| Tool you use | Mind that works for you |

### 1.4 Example Experience

**Monday 7:00 AM - Push Notification:**
```
Good morning. Here's what matters across your worlds:

📍 REAL ESTATE
John Smith listed a competing property at $100K below yours.
I've drafted 3 pricing response strategies.

🚀 STARTUP
Y Combinator W25 applications opened. Given your traction,
you'd be competitive. Deadline: 6 weeks. Want me to start the app?

📱 BRAND X
Your interior design client's competitor went viral with a
room makeover video. I've drafted a similar concept for Brand X.

🔗 CROSS-WORLD
Your slow-selling Austin listing + Brand X content need =
potential collab? They stage it, you both get content.

Tap any item to dive in.
```

---

## 2. The World System

### 2.1 Core Concept

Users don't have one life - they have multiple **Worlds** they operate in:
- A real estate agent in Austin
- A tech startup founder
- A social media manager for multiple brands
- A parent, a hobbyist, a community member

Each World has its own:
- **Goals**: What success looks like
- **Entities**: People, companies, topics that matter
- **Signals**: Information sources to monitor
- **Patterns**: What's worked before

### 2.2 Why Worlds Matter

**Without Worlds:**
- Aria treats all information equally
- No way to prioritize what matters
- Generic, one-size-fits-all responses
- Can't connect insights across domains

**With Worlds:**
- Aria knows WHAT matters in each context
- Filters signals through relevance
- Provides domain-specific intelligence
- Discovers cross-world opportunities

### 2.3 World Definition

A World is a **mental model** of a domain the user operates in:

```yaml
world:
  id: "real-estate-austin"
  name: "Real Estate - Austin Luxury"
  description: "My real estate business focusing on luxury properties in Austin"

  # What does success look like?
  goals:
    - description: "Close 10 deals this quarter"
      priority: high
      deadline: "2026-03-31"
      progress_indicators:
        - "New listings signed"
        - "Properties under contract"
        - "Closed transactions"
      risk_indicators:
        - "Listings expiring"
        - "Deals falling through"

    - description: "Dominate West Austin luxury market"
      priority: high
      progress_indicators:
        - "Market share increase"
        - "Brand recognition"

  # Who/what matters?
  entities:
    - name: "John Smith"
      type: person
      relationship: competitor
      importance: 0.9
      watch_for:
        - "New listings"
        - "Price changes"
        - "Marketing campaigns"
      notes: "Main competitor in luxury segment. Aggressive marketer."

    - name: "Austin Luxury Market"
      type: topic
      relationship: market
      importance: 0.8
      watch_for:
        - "Price trends"
        - "Inventory levels"
        - "Interest rate impacts"

    - name: "Sarah Johnson"
      type: person
      relationship: client
      importance: 0.7
      watch_for:
        - "Ready to buy"
        - "Budget changes"

  # Where to look for signals?
  information_sources:
    - "zillow"
    - "redfin"
    - "mls"
    - "austin business journal"
    - "local news"

  # Keywords that indicate this world is relevant
  keywords:
    - "listing"
    - "property"
    - "real estate"
    - "buyer"
    - "seller"
    - "escrow"
    - "west austin"
    - "luxury home"

  # When is this world typically active?
  schedule:
    active_days: ["monday", "tuesday", "wednesday"]
    active_hours: "9:00-18:00"

  # Learned patterns
  successful_approaches:
    - "Personal video tours get 3x engagement"
    - "Price reductions on day 21 work best"

  failure_patterns:
    - "Generic listing descriptions underperform"
```

### 2.4 World Learning

Worlds aren't just configured - they're **learned**:

**Explicit Learning:**
```
User: "Aria, I also run a social media agency for a few brands."
Aria: "Got it. Let me set up a new world for that.
       What brands do you manage?"
User: "Brand X and Brand Y."
Aria: "What are your main goals for Brand X?"
```

**Observational Learning:**
```
Observation: User spends 3 hours on Zillow/Redfin Mon-Wed
Observation: User mentions "listing" and "buyer" frequently
Observation: User has meetings with "John Smith" marked as competitor

Aria: "I noticed you work on real estate Mon-Wed, focusing on
       luxury properties. John Smith seems to be a competitor.
       Should I set up monitoring for his activity?"
```

**Refinement Learning:**
```
Aria surfaces an insight → User ignores it
Aria surfaces another insight → User acts immediately

Aria learns: Second type of insight is more valuable
Aria adjusts: Prioritize similar insights higher
```

---

## 3. Architecture Overview

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER'S WORLDS                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │ Real Estate │  │ Tech        │  │ Social      │  │ Personal    │   │
│  │             │  │ Startup     │  │ Agency      │  │             │   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘   │
│         └─────────────┬──┴──────────────┬─┘────────────────┘          │
│                       │                 │                              │
│                       ▼                 ▼                              │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                         CORTEX                                   │  │
│  │                   (Synthesis Layer)                              │  │
│  │                                                                  │  │
│  │   • Relevance scoring against worlds                            │  │
│  │   • Priority calculation                                        │  │
│  │   • Cross-world connection detection                            │  │
│  │   • Insight generation                                          │  │
│  │                                                                  │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│         ▲                                         │                    │
│         │                                         ▼                    │
│  ┌──────┴──────┐                         ┌─────────────┐              │
│  │   SIGNALS   │                         │   ACTORS    │              │
│  │  (Watchers) │                         │   (Hands)   │              │
│  │             │                         │             │              │
│  │ • News      │                         │ • Drafts    │              │
│  │ • Social    │                         │ • Responses │              │
│  │ • Market    │                         │ • Schedules │              │
│  │ • Calendar  │                         │ • Research  │              │
│  │ • Email     │                         │             │              │
│  │ • Screen    │                         └──────┬──────┘              │
│  └─────────────┘                                │                     │
│                                                 ▼                     │
│                                        ┌─────────────┐                │
│                                        │  DELIVERY   │                │
│                                        │             │                │
│                                        │ • Voice     │                │
│                                        │ • Push      │                │
│                                        │ • Digest    │                │
│                                        │ • Desktop   │                │
│                                        └─────────────┘                │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

```
1. SIGNAL COLLECTION
   External Sources → Watchers → Raw Signals

2. RELEVANCE FILTERING
   Raw Signals + User Worlds → Cortex → Relevant Signals

3. INSIGHT GENERATION
   Relevant Signals + Goals + History → Cortex → Insights

4. ACTION PREPARATION
   Insights → Actors → Prepared Actions (drafts, responses)

5. DELIVERY DECISION
   Prepared Actions + User Context → Delivery Engine → Right Time/Channel
```

### 3.3 The Ambient Loop

The system runs continuously in the background:

```python
async def ambient_loop():
    while True:
        # 1. Collect signals from all watchers
        signals = await collect_signals(watchers)

        # 2. Filter through user's worlds
        relevant = filter_by_relevance(signals, user.worlds)

        # 3. Generate insights
        insights = generate_insights(relevant, user.goals)

        # 4. Prepare actions for high-priority insights
        for insight in insights:
            if insight.priority > THRESHOLD:
                action = await prepare_action(insight)
                delivery_queue.add(action)

        # 5. Check delivery queue for right-time items
        await check_and_deliver(delivery_queue, user.context)

        # 6. Sleep until next cycle
        await sleep(CHECK_INTERVAL)
```

---

## 4. Core Components

### 4.1 World Manager

**Purpose:** Manages the user's mental models (Worlds)

**Responsibilities:**
- Create, update, delete worlds
- Learn worlds from observation and conversation
- Determine which world(s) are currently active
- Provide world context for signal filtering

**Key Methods:**
```python
class WorldManager:
    def create_world(name: str, description: str) -> World
    def add_goal(world_id: str, goal: Goal) -> bool
    def add_entity(world_id: str, entity: Entity) -> bool
    def get_active_worlds() -> List[World]  # Based on time, context
    def get_relevant_worlds(signal: Signal) -> List[WorldMatch]
    def learn_from_observation(activity: UserActivity) -> None
    def suggest_new_world(observations: List) -> Optional[WorldSuggestion]
```

### 4.2 Signal Watchers

**Purpose:** Collect signals from various sources

**Types of Watchers:**
| Watcher | Sources | Signals |
|---------|---------|---------|
| NewsWatcher | RSS, News APIs | Industry news, trends |
| SocialWatcher | Twitter, LinkedIn, etc. | Mentions, engagement, competitor posts |
| MarketWatcher | Financial APIs | Price changes, market movements |
| CalendarWatcher | Calendar APIs | Events, meetings, deadlines |
| EmailWatcher | Email (optional) | Important messages, action items |
| ScreenWatcher | Local screen | Current context, app usage |

**Key Interface:**
```python
class Watcher(ABC):
    name: str
    check_interval: int  # seconds

    @abstractmethod
    async def observe() -> List[Signal]

    def configure(settings: dict) -> None
```

### 4.3 Cortex (Synthesis Layer)

**Purpose:** Turn raw signals into actionable insights

**Responsibilities:**
- Score signal relevance against user's worlds
- Calculate priority (urgency × relevance × actionability)
- Detect patterns across signals
- Find cross-world connections
- Generate human-readable insights

**Key Methods:**
```python
class Cortex:
    def process_signals(signals: List[Signal]) -> List[Insight]
    def score_relevance(signal: Signal, world: World) -> float
    def calculate_priority(signal: Signal) -> float
    def find_connections(signals: List[Signal]) -> List[Connection]
    def generate_insight(signal: Signal, context: dict) -> Insight
```

### 4.4 Actors (Action Preparers)

**Purpose:** Prepare actions for insights without executing

**Types of Actors:**
| Actor | Prepares |
|-------|----------|
| ContentDrafter | Social posts, threads, articles |
| ResponsePreparer | Replies, DMs, emails |
| ResearchBriefer | Research summaries, briefing docs |
| ScheduleOptimizer | Calendar suggestions, time blocks |
| AlertComposer | Notification content |

**Key Interface:**
```python
class Actor(ABC):
    @abstractmethod
    async def prepare(insight: Insight) -> PreparedAction
```

### 4.5 Delivery Engine

**Purpose:** Deliver insights at the right time via the right channel

**Responsibilities:**
- Determine optimal delivery timing
- Select appropriate channel
- Respect user's focus/DND status
- Batch low-priority items into digests
- Track delivery success

**Delivery Channels:**
| Channel | Use Case | Priority Level |
|---------|----------|----------------|
| Push Notification | Urgent, time-sensitive | Critical |
| Voice Briefing | When Aria is activated | Any |
| Desktop Popup | User is at computer | Medium-High |
| Morning Digest | Daily summary | Low-Medium |
| Email Summary | Weekly recap | Low |

**Key Methods:**
```python
class DeliveryEngine:
    def queue_for_delivery(action: PreparedAction) -> None
    def check_delivery_conditions(action: PreparedAction) -> bool
    def select_channel(action: PreparedAction, context: UserContext) -> Channel
    def deliver(action: PreparedAction, channel: Channel) -> bool
    def compile_digest(period: str) -> Digest
```

### 4.6 Consciousness Stream

**Purpose:** Maintain Aria's running internal state

**Concept:** Aria has a continuous "train of thought" about the user's world. When queried, she doesn't search - she already knows.

**Key Features:**
- Current focus areas
- Active insights being tracked
- Pending preparations
- Recent observations

**Interface:**
```python
class ConsciousnessStream:
    current_thoughts: List[Thought]
    attention_focus: List[str]

    def get_briefing() -> str  # "What are you thinking about?"
    def get_focus() -> str     # "What's top of mind?"
    def log_thought(thought: Thought) -> None
```

---

## 5. Data Models

### 5.1 World

```python
@dataclass
class World:
    id: str
    name: str
    description: str

    # What matters
    goals: List[Goal]
    entities: List[Entity]

    # Where to look
    information_sources: List[str]
    keywords: List[str]

    # Patterns
    schedule: Optional[Schedule]
    successful_approaches: List[str]
    failure_patterns: List[str]

    # Metadata
    created_at: datetime
    updated_at: datetime
    last_active: datetime

    # Learning
    confidence: float  # How well does Aria understand this world?
```

### 5.2 Goal

```python
@dataclass
class Goal:
    id: str
    world_id: str
    description: str
    priority: str  # "critical", "high", "medium", "low"

    # Tracking
    progress_indicators: List[str]
    risk_indicators: List[str]
    deadline: Optional[datetime]

    # Status
    status: str  # "active", "achieved", "paused", "abandoned"
    progress: float  # 0.0 to 1.0
```

### 5.3 Entity

```python
@dataclass
class Entity:
    id: str
    world_id: str
    name: str
    type: str  # "person", "company", "topic", "location", "brand"
    relationship: str  # "competitor", "client", "partner", "target", etc.

    # Monitoring
    importance: float  # 0.0 to 1.0
    watch_for: List[str]  # Events/changes to monitor

    # Context
    notes: str
    last_activity: Optional[datetime]
```

### 5.4 Signal

```python
@dataclass
class Signal:
    id: str
    source: str  # Which watcher
    type: str    # "news", "mention", "price_change", "event", etc.

    # Content
    title: str
    content: str
    url: Optional[str]

    # Relevance (computed)
    relevant_worlds: List[WorldMatch]
    matched_entities: List[str]
    matched_keywords: List[str]

    # Timing
    timestamp: datetime
    expires_at: Optional[datetime]

    # Metadata
    raw_data: dict
```

### 5.5 Insight

```python
@dataclass
class Insight:
    id: str
    signals: List[Signal]  # Source signals

    # Content
    title: str
    summary: str
    world_id: str

    # Priority
    priority: float  # 0.0 to 1.0
    urgency: str     # "immediate", "today", "this_week", "whenever"

    # Suggested action
    suggested_action: str
    action_type: str  # "respond", "draft", "research", "monitor", "ignore"

    # Connections
    related_goals: List[str]
    related_entities: List[str]
    cross_world_connections: List[Connection]

    # Status
    status: str  # "new", "preparing", "ready", "delivered", "acted", "dismissed"
    prepared_action: Optional[PreparedAction]
```

### 5.6 PreparedAction

```python
@dataclass
class PreparedAction:
    id: str
    insight_id: str

    # What's prepared
    type: str  # "draft", "response", "brief", "alert"
    content: str
    options: List[str]  # Alternative versions

    # Quick actions
    one_click_actions: List[QuickAction]

    # Delivery
    preferred_channel: str
    delivery_window: Tuple[datetime, datetime]

    # Status
    status: str  # "prepared", "queued", "delivered", "executed", "expired"
```

---

## 6. User Experience

### 6.1 Onboarding Flow

**Step 1: World Discovery**
```
Aria: "Let's set up my understanding of your work.
       Tell me about the different areas you operate in.
       These could be businesses, projects, or roles."

User: "I'm a real estate agent, I'm building a startup,
       and I run social media for a few brands."

Aria: "Got it - three worlds. Let's set each one up."
```

**Step 2: Goal Setting (per world)**
```
Aria: "For your real estate work - what are you trying
       to achieve right now?"

User: "Close 10 deals this quarter. Beat my competitor John."

Aria: "Who's John? I'll monitor his activity."

User: "John Smith, he's the top luxury agent in West Austin."

Aria: "I'll watch for his listings, price changes, and marketing.
       What else matters in your real estate world?"
```

**Step 3: Source Configuration**
```
Aria: "Where do you get information about real estate?
       I can monitor news sources, social accounts, or market data."

User: "Austin Business Journal, Zillow, and the MLS."

Aria: "I'll set up monitoring for those. I'll also watch for
       Austin real estate news more broadly."
```

### 6.2 Daily Experience

**Morning Briefing (7:00 AM)**
- Push notification with summary
- "Tap to expand" for each world
- Quick actions available

**Throughout the Day**
- Urgent items pushed immediately
- Non-urgent items queued
- Respects focus/DND modes

**On Activation ("Hey Aria")**
- Full verbal briefing
- "What's top of mind?"
- Prepared actions ready

**Evening (Optional)**
- "Here's what happened today"
- Reflection prompts
- Tomorrow preview

### 6.3 Interaction Patterns

**Checking In:**
```
User: "Aria, what's going on?"
Aria: [Delivers current briefing from consciousness stream]
```

**Diving Deep:**
```
User: "Tell me more about John Smith's new listing."
Aria: [Provides detailed analysis, comparisons, suggestions]
```

**Taking Action:**
```
Aria: "I have a response ready for that journalist. Want to send it?"
User: "Read it to me first."
Aria: [Reads draft]
User: "Change the second paragraph to sound more casual."
Aria: [Adjusts and re-reads]
User: "Send it."
```

**Cross-World Query:**
```
User: "Anything connecting my different businesses?"
Aria: [Surfaces cross-world opportunities]
```

---

## 7. Technical Requirements

### 7.1 Infrastructure

**Local Processing:**
- Worlds, goals, entities stored locally
- Signals cached locally
- Privacy-first architecture

**Cloud Dependencies:**
- LLM for synthesis (Gemini/Claude)
- External APIs for signal sources
- Push notification service

**Storage:**
```
~/.aria/
├── ambient/
│   ├── worlds/
│   │   ├── real-estate-austin.yaml
│   │   ├── tech-startup.yaml
│   │   └── social-agency-brandx.yaml
│   ├── watchers/
│   │   ├── news/
│   │   ├── social/
│   │   └── market/
│   ├── cortex/
│   │   ├── insights/
│   │   └── connections/
│   ├── actors/
│   │   ├── drafts/
│   │   └── responses/
│   ├── delivery/
│   │   ├── queue.json
│   │   └── history/
│   └── stream/
│       └── thoughts.log
├── memory/
└── learning/
```

### 7.2 Performance Requirements

| Metric | Target |
|--------|--------|
| Signal collection latency | < 60 seconds |
| Insight generation | < 5 seconds |
| Action preparation | < 10 seconds |
| Voice briefing start | < 2 seconds |
| Background CPU usage | < 5% |
| Background memory | < 200MB |

### 7.3 API Integrations

**Phase 1 (MVP):**
- News: RSS feeds, NewsAPI
- Calendar: Google Calendar
- Web: Generic scraping

**Phase 2:**
- Social: Twitter/X API (or scraping)
- Professional: LinkedIn
- Market: Financial APIs

**Phase 3:**
- Email: Gmail/Outlook (optional, privacy-sensitive)
- Messaging: Slack, Discord
- Custom: User-defined webhooks

### 7.4 Privacy & Security

**Principles:**
- All data stored locally by default
- Cloud processing is stateless
- User controls what's monitored
- Easy data deletion
- Transparency logs

**Sensitive Data Handling:**
- Email content never sent to cloud
- Screen content processed locally when possible
- Financial data encrypted at rest

---

## 8. Implementation Phases

### Phase 1: Foundation (Week 1-2)

**Goal:** Basic world system with manual signals

**Deliverables:**
- [ ] World data model and storage
- [ ] World CRUD operations
- [ ] Goal and entity management
- [ ] Manual signal input
- [ ] Basic relevance scoring

**Files to Create:**
```
aria/ambient/
├── __init__.py
├── models.py          # Data models
├── world_manager.py   # World CRUD
├── storage.py         # YAML/JSON persistence
└── relevance.py       # Relevance scoring
```

### Phase 2: Watchers (Week 2-3)

**Goal:** Automated signal collection

**Deliverables:**
- [ ] Watcher base class
- [ ] NewsWatcher (RSS)
- [ ] CalendarWatcher
- [ ] ScreenContextWatcher
- [ ] Watcher scheduler

**Files to Create:**
```
aria/ambient/watchers/
├── __init__.py
├── base.py            # Watcher ABC
├── news.py            # News/RSS watcher
├── calendar.py        # Calendar watcher
├── screen.py          # Screen context
└── scheduler.py       # Background scheduling
```

### Phase 3: Cortex (Week 3-4)

**Goal:** Intelligent synthesis

**Deliverables:**
- [ ] Signal processing pipeline
- [ ] Priority calculation
- [ ] Insight generation
- [ ] Cross-world connection detection
- [ ] Consciousness stream

**Files to Create:**
```
aria/ambient/
├── cortex.py          # Main synthesis
├── priority.py        # Priority scoring
├── insights.py        # Insight generation
├── connections.py     # Cross-world detection
└── consciousness.py   # Running state
```

### Phase 4: Actors (Week 4-5)

**Goal:** Action preparation

**Deliverables:**
- [ ] Actor base class
- [ ] ContentDrafter
- [ ] ResponsePreparer
- [ ] AlertComposer
- [ ] Integration with existing tools

**Files to Create:**
```
aria/ambient/actors/
├── __init__.py
├── base.py            # Actor ABC
├── content.py         # Content drafting
├── response.py        # Response preparation
└── alert.py           # Alert composition
```

### Phase 5: Delivery (Week 5-6)

**Goal:** Right-time delivery

**Deliverables:**
- [ ] Delivery engine
- [ ] Channel selection logic
- [ ] Digest compilation
- [ ] Voice briefing integration
- [ ] (Optional) Push notification setup

**Files to Create:**
```
aria/ambient/
├── delivery.py        # Delivery engine
├── channels.py        # Channel definitions
└── digest.py          # Digest compilation
```

### Phase 6: Learning & Refinement (Week 6-7)

**Goal:** Self-improving system

**Deliverables:**
- [ ] Feedback collection
- [ ] World inference from behavior
- [ ] Priority model training
- [ ] Automatic entity discovery

**Files to Modify:**
```
aria/ambient/
├── world_manager.py   # Add learning
├── cortex.py          # Add feedback loop
└── learning.py        # NEW: Learning logic
```

### Phase 7: Integration & Polish (Week 7-8)

**Goal:** Production-ready system

**Deliverables:**
- [ ] Integration with main Aria
- [ ] Performance optimization
- [ ] Error handling
- [ ] Documentation
- [ ] User testing

---

## 9. Success Metrics

### 9.1 Engagement Metrics

| Metric | Target |
|--------|--------|
| Daily briefing open rate | > 80% |
| Insights acted upon | > 30% |
| Prepared actions used | > 50% |
| User-initiated queries | Decreasing (proactive taking over) |

### 9.2 Quality Metrics

| Metric | Target |
|--------|--------|
| Relevant insights (user rating) | > 4/5 |
| Cross-world connections found | > 2/week |
| False positive rate | < 10% |
| Missed important signals | < 5% |

### 9.3 Performance Metrics

| Metric | Target |
|--------|--------|
| System uptime | > 99% |
| Briefing generation time | < 5s |
| Background resource usage | < 5% CPU |

---

## 10. Open Questions

### 10.1 Technical

1. **API Access:** How do we reliably access social media data given API restrictions?
   - Option A: Official APIs (expensive, limited)
   - Option B: Browser automation (fragile)
   - Option C: User provides access tokens
   - Option D: Focus on non-restricted sources first

2. **LLM Costs:** Continuous synthesis is expensive. How do we optimize?
   - Option A: Local models for filtering, cloud for synthesis
   - Option B: Aggressive caching
   - Option C: Tiered processing

3. **Push Notifications:** Mobile app required for true ambient experience?
   - Option A: Build simple mobile companion app
   - Option B: Use Telegram/SMS as bridge
   - Option C: Desktop-only MVP

### 10.2 Product

1. **Information Overload:** How do we ensure Aria doesn't become another notification spam source?
   - Strong filtering
   - User-adjustable thresholds
   - "Quiet mode" options

2. **Privacy Balance:** Users want monitoring but also privacy. Where's the line?
   - Clear opt-in for each source
   - Easy way to see what's being collected
   - Simple "forget" functionality

3. **Multi-User:** What if multiple family members use Aria?
   - Separate profiles
   - Shared vs private worlds

### 10.3 Business

1. **Monetization:** If this is valuable, how does it generate revenue?
   - Premium features
   - API access limits
   - B2B licensing

2. **Differentiation:** What stops others from copying this?
   - Deep integration with existing Aria
   - Learning/personalization over time
   - Open source community

---

## Appendix A: Example World Configurations

### A.1 Real Estate Agent

```yaml
id: real-estate-austin
name: "Real Estate - Austin Luxury"
description: "My real estate business in the Austin luxury market"

goals:
  - description: "Close 10 deals this quarter"
    priority: high
    deadline: "2026-03-31"
    progress_indicators:
      - "Properties under contract"
      - "Listings signed"
    risk_indicators:
      - "Listings expiring"
      - "Deals falling through"

  - description: "Become #1 in West Austin luxury"
    priority: high
    progress_indicators:
      - "Market share increase"
      - "Referral rate"

entities:
  - name: "John Smith"
    type: person
    relationship: competitor
    importance: 0.9
    watch_for: ["new listings", "price changes", "marketing"]

  - name: "West Austin Market"
    type: topic
    importance: 0.8
    watch_for: ["price trends", "inventory", "new developments"]

information_sources:
  - "zillow"
  - "redfin"
  - "austin business journal"
  - "statesman real estate"

keywords:
  - "austin real estate"
  - "luxury home"
  - "west austin"
  - "listing"
  - "open house"
```

### A.2 Tech Startup Founder

```yaml
id: startup-aria
name: "Aria Startup"
description: "Building Aria - the ambient intelligence assistant"

goals:
  - description: "Launch MVP by Q3"
    priority: critical
    deadline: "2026-09-30"
    progress_indicators:
      - "Features shipped"
      - "Beta users"
    risk_indicators:
      - "Technical blockers"
      - "Resource constraints"

  - description: "Raise seed round"
    priority: high
    progress_indicators:
      - "Investor meetings"
      - "Term sheets"

entities:
  - name: "Anthropic"
    type: company
    relationship: technology_partner
    importance: 0.9
    watch_for: ["API updates", "new features", "pricing changes"]

  - name: "AI Agent Market"
    type: topic
    importance: 0.8
    watch_for: ["competitor launches", "market trends", "funding news"]

information_sources:
  - "hacker news"
  - "techcrunch"
  - "twitter ai"
  - "anthropic blog"

keywords:
  - "ai agent"
  - "voice assistant"
  - "ambient computing"
  - "claude"
  - "gemini"
```

### A.3 Social Media Manager

```yaml
id: social-agency-brandx
name: "Social Agency - Brand X"
description: "Managing social media for Brand X (interior design)"

goals:
  - description: "Grow to 50K followers"
    priority: high
    deadline: "2026-06-30"
    progress_indicators:
      - "Follower growth rate"
      - "Viral posts"
    risk_indicators:
      - "Engagement dropping"
      - "Negative sentiment"

  - description: "5% engagement rate"
    priority: medium
    progress_indicators:
      - "Likes/comments per post"
      - "Share rate"

entities:
  - name: "Brand X"
    type: brand
    relationship: client
    importance: 1.0
    watch_for: ["mentions", "sentiment", "competitor activity"]

  - name: "Interior Design Trends"
    type: topic
    importance: 0.7
    watch_for: ["viral content", "seasonal trends", "influencer posts"]

information_sources:
  - "instagram design"
  - "pinterest home"
  - "design blogs"

keywords:
  - "interior design"
  - "home decor"
  - "brand x"
  - "design trends"
```

---

## Appendix B: Signal Types

| Signal Type | Description | Sources |
|-------------|-------------|---------|
| news_article | News about relevant topic | NewsAPI, RSS |
| social_mention | Brand/entity mentioned | Twitter, LinkedIn |
| social_post | Post by tracked entity | Twitter, LinkedIn |
| price_change | Market price movement | Financial APIs |
| calendar_event | Upcoming event | Google Calendar |
| competitor_activity | Competitor did something | Various |
| trend_emerging | Topic gaining traction | Social, News |
| deadline_approaching | Goal deadline coming | Internal |
| engagement_anomaly | Unusual engagement | Social |
| opportunity_detected | Cross-world connection | Internal |

---

## Appendix C: Priority Calculation

```python
def calculate_priority(signal: Signal, world: World) -> float:
    """
    Priority = Urgency × Relevance × Actionability × Novelty

    Each factor is 0.0 to 1.0
    Final priority is 0.0 to 1.0
    """

    # Urgency: How time-sensitive is this?
    urgency = calculate_urgency(signal)
    # - Breaking news: 1.0
    # - Today relevant: 0.7
    # - This week: 0.4
    # - Whenever: 0.2

    # Relevance: How much does this matter to user's goals?
    relevance = calculate_relevance(signal, world)
    # - Directly impacts goal: 1.0
    # - Related to entity: 0.7
    # - Keyword match: 0.4
    # - Source match: 0.3

    # Actionability: Can user do something about this?
    actionability = calculate_actionability(signal)
    # - Clear action available: 1.0
    # - Requires research: 0.6
    # - FYI only: 0.3

    # Novelty: Is this new information?
    novelty = calculate_novelty(signal)
    # - Never seen: 1.0
    # - Update to known: 0.5
    # - Duplicate: 0.0

    return urgency * relevance * actionability * novelty
```

---

*End of PRD*
