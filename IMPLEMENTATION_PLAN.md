# Aria Enhancement Implementation Plan

> Parallel implementation of 5 major features
> Created: 2026-01-15

---

## Feature 1: MCP Integration

### Goal
Add official MCP servers (filesystem, fetch, git) to give Aria 60+ new tools without writing code.

### Files to Create/Modify
- `aria/mcp_client.py` (NEW) - MCP client to call external servers
- `aria/config.py` (MODIFY) - Add MCP server configurations
- `aria/agent.py` (MODIFY) - Integrate MCP tools into agent loop
- `~/.aria/mcp_servers.json` (NEW) - User MCP server config

### Implementation Steps

#### Step 1.1: Create MCP Client
```python
# aria/mcp_client.py
class MCPClient:
    """Client for calling external MCP servers."""

    def __init__(self, server_config: dict):
        self.servers = {}  # name -> subprocess

    def start_server(self, name: str, command: str, args: list):
        """Start an MCP server subprocess."""

    def call_tool(self, server: str, tool: str, arguments: dict):
        """Call a tool on an MCP server via JSON-RPC."""

    def list_tools(self, server: str) -> list:
        """Get available tools from a server."""

    def stop_all(self):
        """Shutdown all server subprocesses."""
```

#### Step 1.2: Configure Default Servers
```python
# In config.py
MCP_SERVERS = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", str(Path.home())],
        "description": "File operations"
    },
    "fetch": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-fetch"],
        "description": "Web content fetching"
    },
    "git": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-git"],
        "description": "Git operations"
    }
}
```

#### Step 1.3: Integrate into Agent
- Add `self.mcp_client = MCPClient(config)` to AriaAgent.__init__
- Add MCP tools to tool schema for Claude
- Route MCP tool calls through mcp_client.call_tool()

### Success Criteria
- [ ] `filesystem` server starts and responds to list_tools
- [ ] Can read/write files via MCP
- [ ] Can fetch web content via MCP
- [ ] Tools appear in agent's available actions

---

## Feature 2: OpenAI Realtime Voice

### Goal
Replace Whisper STT → Claude → TTS pipeline with OpenAI Realtime API for <800ms latency.

### Files to Create/Modify
- `aria/realtime_voice.py` (NEW) - Realtime API client
- `aria/voice.py` (MODIFY) - Add realtime mode option
- `aria/config.py` (MODIFY) - Add OPENAI_API_KEY, voice settings
- `aria/agent.py` (MODIFY) - Support realtime voice mode

### Implementation Steps

#### Step 2.1: Create Realtime Client
```python
# aria/realtime_voice.py
import websockets
import json
import base64
import pyaudio

class RealtimeVoiceClient:
    """OpenAI Realtime API client for voice interaction."""

    REALTIME_URL = "wss://api.openai.com/v1/realtime"

    def __init__(self, api_key: str, model: str = "gpt-4o-realtime-preview"):
        self.api_key = api_key
        self.model = model
        self.ws = None
        self.audio = pyaudio.PyAudio()
        self.session_config = {
            "modalities": ["text", "audio"],
            "voice": "alloy",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "silence_duration_ms": 500
            }
        }

    async def connect(self):
        """Establish WebSocket connection."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        self.ws = await websockets.connect(
            f"{self.REALTIME_URL}?model={self.model}",
            extra_headers=headers
        )
        await self._configure_session()

    async def _configure_session(self):
        """Send session configuration."""
        await self.ws.send(json.dumps({
            "type": "session.update",
            "session": self.session_config
        }))

    async def send_audio(self, audio_chunk: bytes):
        """Send audio chunk to API."""
        await self.ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio_chunk).decode()
        }))

    async def receive_events(self):
        """Generator for receiving events."""
        async for message in self.ws:
            event = json.loads(message)
            yield event

    async def add_tool(self, name: str, description: str, parameters: dict):
        """Add a tool for function calling."""
        await self.ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "tools": [{
                    "type": "function",
                    "name": name,
                    "description": description,
                    "parameters": parameters
                }]
            }
        }))

    async def send_tool_result(self, call_id: str, result: str):
        """Send tool call result back."""
        await self.ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result
            }
        }))
        await self.ws.send(json.dumps({"type": "response.create"}))
```

#### Step 2.2: Audio Streaming
```python
class AudioStreamer:
    """Handles microphone input and speaker output."""

    def __init__(self, sample_rate=24000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = 1024

    def start_recording(self, callback):
        """Start streaming from microphone."""

    def play_audio(self, audio_data: bytes):
        """Play audio through speakers."""

    def handle_interruption(self):
        """Stop playback when user interrupts."""
```

#### Step 2.3: Integration
- Add `RealtimeVoiceClient` as alternative to current voice system
- Support both modes: `whisper` (current) and `realtime` (new)
- Add Aria's tools to realtime session
- Handle function calls from realtime API

### Success Criteria
- [ ] WebSocket connection established
- [ ] Audio streams bidirectionally
- [ ] Voice-to-voice latency <1 second
- [ ] Function calling works (click, type, etc.)
- [ ] Interruption handling works

---

## Feature 3: MediaPipe Gesture Control

### Goal
Add gesture recognition for hands-free confirmation and control.

### Files to Create/Modify
- `aria/gestures.py` (NEW) - MediaPipe gesture recognition
- `aria/config.py` (MODIFY) - Gesture settings
- `aria/agent.py` (MODIFY) - Integrate gesture input
- `requirements.txt` (MODIFY) - Add mediapipe

### Implementation Steps

#### Step 3.1: Create Gesture Recognizer
```python
# aria/gestures.py
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum

class Gesture(Enum):
    NONE = "none"
    THUMBS_UP = "Thumb_Up"
    THUMBS_DOWN = "Thumb_Down"
    OPEN_PALM = "Open_Palm"
    CLOSED_FIST = "Closed_Fist"
    POINTING_UP = "Pointing_Up"
    VICTORY = "Victory"
    I_LOVE_YOU = "ILoveYou"

@dataclass
class GestureEvent:
    gesture: Gesture
    confidence: float
    handedness: str  # "Left" or "Right"

class GestureRecognizer:
    """MediaPipe-based gesture recognition."""

    # Map gestures to Aria actions
    GESTURE_ACTIONS = {
        Gesture.THUMBS_UP: "confirm",
        Gesture.THUMBS_DOWN: "cancel",
        Gesture.OPEN_PALM: "stop",
        Gesture.CLOSED_FIST: "pause",
        Gesture.POINTING_UP: "select",
    }

    def __init__(self, model_path: Optional[str] = None):
        # Download model if not provided
        self.model_path = model_path or self._download_model()

        # Create recognizer
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self._on_result
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

        self.current_gesture: Optional[GestureEvent] = None
        self.callbacks: list[Callable[[GestureEvent], None]] = []
        self.camera = None
        self.running = False

    def _download_model(self) -> str:
        """Download gesture recognizer model."""
        import urllib.request
        model_url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
        model_path = Path.home() / ".aria" / "models" / "gesture_recognizer.task"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if not model_path.exists():
            urllib.request.urlretrieve(model_url, model_path)
        return str(model_path)

    def _on_result(self, result, output_image, timestamp_ms):
        """Callback for gesture recognition results."""
        if result.gestures and result.gestures[0]:
            gesture = result.gestures[0][0]
            handedness = result.handedness[0][0]

            try:
                gesture_type = Gesture(gesture.category_name)
            except ValueError:
                gesture_type = Gesture.NONE

            event = GestureEvent(
                gesture=gesture_type,
                confidence=gesture.score,
                handedness=handedness.category_name
            )

            # Only trigger if confidence > 0.7
            if event.confidence > 0.7 and event.gesture != Gesture.NONE:
                self.current_gesture = event
                for callback in self.callbacks:
                    callback(event)

    def on_gesture(self, callback: Callable[[GestureEvent], None]):
        """Register a gesture callback."""
        self.callbacks.append(callback)

    def start(self):
        """Start gesture recognition from camera."""
        self.camera = cv2.VideoCapture(0)
        self.running = True

        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                continue

            # Convert to MediaPipe image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # Process frame
            timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            self.recognizer.recognize_async(mp_image, timestamp)

            # Small delay to limit CPU usage
            cv2.waitKey(30)

    def stop(self):
        """Stop gesture recognition."""
        self.running = False
        if self.camera:
            self.camera.release()

    def get_action(self, gesture: Gesture) -> Optional[str]:
        """Get the action associated with a gesture."""
        return self.GESTURE_ACTIONS.get(gesture)
```

#### Step 3.2: Face Presence Detection
```python
class FacePresenceDetector:
    """Detect if user is present (looking at screen)."""

    def __init__(self):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        self.is_present = False
        self.last_seen = None
        self.away_threshold = 5.0  # seconds

    def update(self, frame) -> bool:
        """Update presence status from camera frame."""
        results = self.face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.detections:
            self.is_present = True
            self.last_seen = time.time()
        elif self.last_seen and (time.time() - self.last_seen) > self.away_threshold:
            self.is_present = False

        return self.is_present
```

#### Step 3.3: Integration
- Run gesture recognition in background thread
- Add gesture events to agent input processing
- Use thumbs up for confirmation dialogs
- Wake Aria when face detected, sleep when away

### Success Criteria
- [ ] MediaPipe model downloads and loads
- [ ] Camera captures and processes frames at 30fps
- [ ] Thumbs up triggers confirmation
- [ ] Open palm stops current action
- [ ] Face presence detection works

---

## Feature 4: Hierarchical Memory

### Goal
Upgrade memory from flat ChromaDB to hierarchical structure with multi-index search.

### Files to Create/Modify
- `aria/memory_v2.py` (NEW) - Hierarchical memory system
- `aria/memory.py` (MODIFY) - Deprecate, delegate to v2
- `aria/indices.py` (NEW) - Multi-index search (BM25, vector, time)
- `~/.aria/data/memory_v2/` (NEW) - New storage structure

### Implementation Steps

#### Step 4.1: Define Memory Schema
```python
# aria/memory_v2.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import json

class MemoryType(Enum):
    FACT = "fact"           # User facts/preferences
    EVENT = "event"         # Things that happened
    COMMAND = "command"     # Learned voice commands
    PROCEDURE = "procedure" # Multi-step workflows
    CONTEXT = "context"     # Current project/task context

@dataclass
class MemoryItem:
    """Single memory item with full metadata."""
    id: str
    content: str
    memory_type: MemoryType
    category: str  # e.g., "preference", "personal", "work"

    # Quality metrics
    confidence: float = 0.8
    usage_count: int = 0
    success_rate: float = 1.0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None

    # Relationships
    parent_id: Optional[str] = None  # For hierarchical grouping
    related_ids: List[str] = field(default_factory=list)

    # Source tracking
    source: str = "user"  # "user", "inferred", "system"
    source_context: Optional[str] = None  # Original conversation/event

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "type": self.memory_type.value,
            "category": self.category,
            "confidence": self.confidence,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
        }

@dataclass
class MemoryCategory:
    """Category-level summary of memories."""
    name: str
    description: str
    item_count: int = 0
    summary: str = ""  # LLM-generated summary
    keywords: List[str] = field(default_factory=list)
    updated_at: datetime = field(default_factory=datetime.now)
```

#### Step 4.2: Create Multi-Index Search
```python
# aria/indices.py
from rank_bm25 import BM25Okapi
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple
import hashlib

class MultiIndexSearch:
    """Combined search across multiple index types."""

    def __init__(self, embedding_model=None):
        self.items: List[MemoryItem] = []
        self.bm25_index = None
        self.vector_index = None  # ChromaDB collection
        self.time_index: Dict[str, datetime] = {}
        self.simhash_index: Dict[int, str] = {}  # For deduplication
        self.embedding_model = embedding_model

    def add(self, item: MemoryItem):
        """Add item to all indices."""
        self.items.append(item)
        self._update_bm25()
        self._update_vector(item)
        self._update_time(item)
        self._update_simhash(item)

    def _update_bm25(self):
        """Rebuild BM25 index."""
        tokenized = [item.content.lower().split() for item in self.items]
        self.bm25_index = BM25Okapi(tokenized)

    def _update_vector(self, item: MemoryItem):
        """Add to vector index."""
        if self.vector_index:
            self.vector_index.add(
                ids=[item.id],
                documents=[item.content],
                metadatas=[item.to_dict()]
            )

    def _update_time(self, item: MemoryItem):
        """Add to time index."""
        self.time_index[item.id] = item.created_at

    def _compute_simhash(self, text: str) -> int:
        """Compute SimHash for deduplication."""
        # Simplified SimHash
        words = text.lower().split()
        hash_bits = 64
        v = [0] * hash_bits

        for word in words:
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
            for i in range(hash_bits):
                if word_hash & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1

        return sum(1 << i for i in range(hash_bits) if v[i] > 0)

    def _update_simhash(self, item: MemoryItem):
        """Add to SimHash index for deduplication."""
        simhash = self._compute_simhash(item.content)
        self.simhash_index[simhash] = item.id

    def is_duplicate(self, content: str, threshold: int = 3) -> bool:
        """Check if content is duplicate (hamming distance < threshold)."""
        new_hash = self._compute_simhash(content)
        for existing_hash in self.simhash_index:
            distance = bin(new_hash ^ existing_hash).count('1')
            if distance < threshold:
                return True
        return False

    def search(
        self,
        query: str,
        n_results: int = 10,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        memory_types: Optional[List[MemoryType]] = None,
        min_confidence: float = 0.3
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Search across all indices with fusion.

        Returns list of (item, score) tuples.
        """
        # BM25 scores
        bm25_scores = self._bm25_search(query)

        # Vector scores
        vector_scores = self._vector_search(query)

        # Time decay (recent items score higher)
        time_scores = self._time_scores()

        # Combine scores (weighted fusion)
        combined = {}
        for item_id, score in bm25_scores.items():
            combined[item_id] = combined.get(item_id, 0) + score * 0.3
        for item_id, score in vector_scores.items():
            combined[item_id] = combined.get(item_id, 0) + score * 0.5
        for item_id, score in time_scores.items():
            combined[item_id] = combined.get(item_id, 0) + score * 0.2

        # Filter and sort
        results = []
        for item in self.items:
            if item.id not in combined:
                continue
            if item.confidence < min_confidence:
                continue
            if memory_types and item.memory_type not in memory_types:
                continue
            if time_range:
                if not (time_range[0] <= item.created_at <= time_range[1]):
                    continue
            results.append((item, combined[item.id]))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n_results]

    def temporal_query(self, description: str) -> List[MemoryItem]:
        """
        Query by time description.

        Examples:
        - "today"
        - "yesterday"
        - "last week"
        - "this morning"
        """
        now = datetime.now()

        if "today" in description.lower():
            start = now.replace(hour=0, minute=0, second=0)
            end = now
        elif "yesterday" in description.lower():
            start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0)
            end = now.replace(hour=0, minute=0, second=0)
        elif "last week" in description.lower():
            start = now - timedelta(weeks=1)
            end = now
        elif "this morning" in description.lower():
            start = now.replace(hour=0, minute=0, second=0)
            end = now.replace(hour=12, minute=0, second=0)
        else:
            return []

        return [
            item for item in self.items
            if start <= item.created_at <= end
        ]
```

#### Step 4.3: Hierarchical Memory Manager
```python
class HierarchicalMemory:
    """
    Three-tier memory system:
    - Categories: High-level summaries
    - Items: Individual memories
    - Resources: Raw data (transcripts, screenshots)
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.categories: Dict[str, MemoryCategory] = {}
        self.search = MultiIndexSearch()

        # ChromaDB for vector search
        self.chroma = chromadb.PersistentClient(path=str(storage_path / "chroma"))
        self.collection = self.chroma.get_or_create_collection("memories_v2")
        self.search.vector_index = self.collection

        self._load()

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        category: str = "general",
        confidence: float = 0.8,
        source_context: Optional[str] = None
    ) -> Optional[MemoryItem]:
        """Store a new memory."""
        # Check for duplicates
        if self.search.is_duplicate(content):
            # Update existing instead
            return self._reinforce_similar(content)

        item = MemoryItem(
            id=self._generate_id(content),
            content=content,
            memory_type=memory_type,
            category=category,
            confidence=confidence,
            source_context=source_context
        )

        self.search.add(item)
        self._update_category(category, item)
        self._save()

        return item

    def recall(
        self,
        query: str,
        n_results: int = 5,
        **kwargs
    ) -> List[Tuple[MemoryItem, float]]:
        """Search memories."""
        return self.search.search(query, n_results, **kwargs)

    def what_happened(self, time_description: str) -> List[MemoryItem]:
        """Temporal query: 'What did I do today?'"""
        return self.search.temporal_query(time_description)

    def get_category_summary(self, category: str) -> str:
        """Get LLM-generated summary of a category."""
        if category in self.categories:
            return self.categories[category].summary
        return ""

    def _update_category(self, category: str, item: MemoryItem):
        """Update category with new item."""
        if category not in self.categories:
            self.categories[category] = MemoryCategory(
                name=category,
                description=f"Memories about {category}"
            )
        self.categories[category].item_count += 1
        self.categories[category].updated_at = datetime.now()
```

### Success Criteria
- [ ] Memories stored in hierarchical structure
- [ ] BM25 keyword search works
- [ ] Vector semantic search works
- [ ] Temporal queries work ("what did I do today?")
- [ ] Duplicate detection prevents redundant storage
- [ ] Category summaries generated

---

## Feature 5: Multi-Agent Handoffs

### Goal
Add specialist agents with Swarm-style routing for better task handling.

### Files to Create/Modify
- `aria/agents/` (NEW) - Agent module
- `aria/agents/base.py` (NEW) - Base agent class
- `aria/agents/coordinator.py` (NEW) - Main routing agent
- `aria/agents/file_agent.py` (NEW) - File operations specialist
- `aria/agents/browser_agent.py` (NEW) - Web/browser specialist
- `aria/agents/system_agent.py` (NEW) - Desktop control specialist
- `aria/agent.py` (MODIFY) - Use coordinator for routing

### Implementation Steps

#### Step 5.1: Define Base Agent
```python
# aria/agents/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Callable, Any

@dataclass
class AgentContext:
    """Context passed between agents."""
    user_input: str
    memory_context: str
    screen_context: Optional[str] = None
    variables: dict = field(default_factory=dict)
    history: List[dict] = field(default_factory=list)

@dataclass
class AgentResult:
    """Result from agent execution."""
    success: bool
    response: str
    handoff_to: Optional[str] = None  # Agent name to hand off to
    data: Optional[dict] = None

class BaseAgent(ABC):
    """Base class for specialist agents."""

    name: str = "base"
    description: str = "Base agent"
    triggers: List[str] = []

    def __init__(self, tools: List[Callable] = None):
        self.tools = tools or []

    @abstractmethod
    async def process(self, context: AgentContext) -> AgentResult:
        """Process a request."""
        pass

    def matches(self, text: str) -> float:
        """Check if this agent should handle the request."""
        text_lower = text.lower()
        for trigger in self.triggers:
            if trigger in text_lower:
                return 0.9
        return 0.0

    def handoff(self, agent_name: str, response: str = "") -> AgentResult:
        """Create a handoff result."""
        return AgentResult(
            success=True,
            response=response,
            handoff_to=agent_name
        )
```

#### Step 5.2: Create Specialist Agents
```python
# aria/agents/file_agent.py
class FileAgent(BaseAgent):
    """Specialist for file operations."""

    name = "file"
    description = "Handles file operations: find, read, write, move, delete"
    triggers = ["file", "find", "search for", "move", "copy", "delete", "create file"]

    async def process(self, context: AgentContext) -> AgentResult:
        # Use MCP filesystem tools
        ...

# aria/agents/browser_agent.py
class BrowserAgent(BaseAgent):
    """Specialist for web and browser tasks."""

    name = "browser"
    description = "Handles web tasks: search, navigate, extract content"
    triggers = ["search", "google", "website", "browse", "url", "web"]

    async def process(self, context: AgentContext) -> AgentResult:
        # Use browser control + MCP fetch
        ...

# aria/agents/system_agent.py
class SystemAgent(BaseAgent):
    """Specialist for desktop control."""

    name = "system"
    description = "Handles desktop control: click, type, open apps, screenshots"
    triggers = ["click", "type", "open", "screenshot", "window"]

    async def process(self, context: AgentContext) -> AgentResult:
        # Use existing control module
        ...

# aria/agents/code_agent.py
class CodeAgent(BaseAgent):
    """Specialist for coding tasks - delegates to Claude Code."""

    name = "code"
    description = "Handles coding: write code, fix bugs, run tests"
    triggers = ["code", "implement", "fix bug", "test", "commit", "git"]

    async def process(self, context: AgentContext) -> AgentResult:
        # Use claude_bridge
        ...
```

#### Step 5.3: Create Coordinator
```python
# aria/agents/coordinator.py
class Coordinator:
    """
    Routes requests to specialist agents.

    Uses Swarm-style handoffs for multi-step tasks.
    """

    def __init__(self):
        self.agents = {
            "file": FileAgent(),
            "browser": BrowserAgent(),
            "system": SystemAgent(),
            "code": CodeAgent(),
        }
        self.default_agent = "system"

    def route(self, context: AgentContext) -> str:
        """Determine which agent should handle the request."""
        scores = {}
        for name, agent in self.agents.items():
            scores[name] = agent.matches(context.user_input)

        # Return highest scoring agent
        best = max(scores, key=scores.get)
        if scores[best] > 0.5:
            return best
        return self.default_agent

    async def process(self, context: AgentContext) -> AgentResult:
        """Process request, handling handoffs."""
        max_handoffs = 3
        current_agent = self.route(context)

        for _ in range(max_handoffs):
            agent = self.agents[current_agent]
            result = await agent.process(context)

            if result.handoff_to and result.handoff_to in self.agents:
                current_agent = result.handoff_to
                context.history.append({
                    "agent": agent.name,
                    "response": result.response
                })
            else:
                return result

        return AgentResult(
            success=False,
            response="Too many handoffs, something went wrong."
        )
```

### Success Criteria
- [ ] Coordinator routes to correct specialist
- [ ] FileAgent handles file operations
- [ ] BrowserAgent handles web tasks
- [ ] SystemAgent handles desktop control
- [ ] CodeAgent delegates to Claude Code
- [ ] Handoffs work between agents

---

## Parallel Execution Plan

### Agent Assignment

| Agent | Feature | Files |
|-------|---------|-------|
| Agent 1 | MCP Integration | mcp_client.py, config.py |
| Agent 2 | Realtime Voice | realtime_voice.py, voice.py |
| Agent 3 | Gesture Control | gestures.py |
| Agent 4 | Hierarchical Memory | memory_v2.py, indices.py |
| Agent 5 | Multi-Agent | agents/*.py |

### Dependencies
- Feature 5 (Multi-Agent) depends on Feature 1 (MCP) for FileAgent
- Feature 2 (Realtime Voice) is independent
- Feature 3 (Gestures) is independent
- Feature 4 (Memory) is independent

### Execution Order
1. Start all 5 agents in parallel
2. Agent 1 (MCP) should complete first (simplest)
3. Agents 2, 3, 4 can complete independently
4. Agent 5 integrates after Agent 1 completes

