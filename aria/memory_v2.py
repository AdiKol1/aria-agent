"""
Hierarchical Memory System for Aria v2.

Three-tier architecture:
- Categories: High-level summaries
- Items: Individual memories
- Resources: Raw data (transcripts, screenshots)

Multi-index search:
- Vector (semantic similarity)
- BM25 (keyword matching)
- Time (temporal queries)
- SimHash (deduplication)
"""

import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import chromadb
from chromadb.config import Settings

# Try to import rank_bm25, install if needed
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "rank-bm25"], check=True)
    from rank_bm25 import BM25Okapi


class MemoryType(Enum):
    """Types of memories."""
    FACT = "fact"
    EVENT = "event"
    COMMAND = "command"
    PROCEDURE = "procedure"
    CONTEXT = "context"
    PREFERENCE = "preference"


@dataclass
class MemoryItem:
    """A single memory item."""
    id: str
    content: str
    memory_type: MemoryType
    category: str

    # Quality metrics
    confidence: float = 0.8
    usage_count: int = 0
    success_rate: float = 1.0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None

    # Relationships
    parent_id: Optional[str] = None
    related_ids: List[str] = field(default_factory=list)

    # Source
    source: str = "user"
    source_context: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "category": self.category,
            "confidence": self.confidence,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "parent_id": self.parent_id,
            "related_ids": self.related_ids,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryItem":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            category=data["category"],
            confidence=data.get("confidence", 0.8),
            usage_count=data.get("usage_count", 0),
            success_rate=data.get("success_rate", 1.0),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            parent_id=data.get("parent_id"),
            related_ids=data.get("related_ids", []),
            source=data.get("source", "user"),
        )


@dataclass
class MemoryCategory:
    """Category-level summary."""
    name: str
    description: str
    item_count: int = 0
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    updated_at: datetime = field(default_factory=datetime.now)


class MultiIndexSearch:
    """Multi-index search engine."""

    def __init__(self):
        self.items: List[MemoryItem] = []
        self.item_index: Dict[str, int] = {}  # id -> index
        self.bm25_index: Optional[BM25Okapi] = None
        self.simhash_index: Dict[int, str] = {}

    def add(self, item: MemoryItem):
        """Add item to all indices."""
        self.items.append(item)
        self.item_index[item.id] = len(self.items) - 1
        self._rebuild_bm25()
        self._add_simhash(item)

    def remove(self, item_id: str):
        """Remove item from indices."""
        if item_id in self.item_index:
            idx = self.item_index[item_id]
            del self.items[idx]
            del self.item_index[item_id]
            # Rebuild indices
            self.item_index = {item.id: i for i, item in enumerate(self.items)}
            self._rebuild_bm25()

    def _rebuild_bm25(self):
        """Rebuild BM25 index."""
        if not self.items:
            self.bm25_index = None
            return
        tokenized = [item.content.lower().split() for item in self.items]
        self.bm25_index = BM25Okapi(tokenized)

    def _compute_simhash(self, text: str, bits: int = 64) -> int:
        """Compute SimHash for deduplication."""
        words = text.lower().split()
        v = [0] * bits

        for word in words:
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            for i in range(bits):
                if h & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1

        return sum(1 << i for i in range(bits) if v[i] > 0)

    def _add_simhash(self, item: MemoryItem):
        """Add to SimHash index."""
        h = self._compute_simhash(item.content)
        self.simhash_index[h] = item.id

    def is_duplicate(self, content: str, threshold: int = 3) -> Optional[str]:
        """Check if content is duplicate. Returns existing item ID if duplicate."""
        new_hash = self._compute_simhash(content)

        for existing_hash, item_id in self.simhash_index.items():
            distance = bin(new_hash ^ existing_hash).count('1')
            if distance < threshold:
                return item_id

        return None

    def bm25_search(self, query: str, n: int = 10) -> List[Tuple[str, float]]:
        """BM25 keyword search."""
        if not self.bm25_index:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)

        # Get top N
        indexed_scores = [(self.items[i].id, scores[i]) for i in range(len(scores))]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores[:n]

    def time_search(self, start: datetime, end: datetime) -> List[MemoryItem]:
        """Search by time range."""
        return [
            item for item in self.items
            if start <= item.created_at <= end
        ]


class HierarchicalMemory:
    """
    Hierarchical memory system with multi-index search.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or (Path.home() / ".aria" / "data" / "memory_v2")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Categories
        self.categories: Dict[str, MemoryCategory] = {}

        # Multi-index search
        self.search = MultiIndexSearch()

        # ChromaDB for vector search
        self.chroma = chromadb.PersistentClient(
            path=str(self.storage_path / "chroma"),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma.get_or_create_collection(
            name="memories_v2",
            metadata={"description": "Hierarchical memory for Aria"}
        )

        # Load existing data
        self._load()

        print(f"Hierarchical Memory initialized: {len(self.search.items)} items in {len(self.categories)} categories")

    def _load(self):
        """Load memories from storage."""
        # Load from ChromaDB
        results = self.collection.get()

        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                try:
                    meta = results["metadatas"][i] if results["metadatas"] else {}
                    item = MemoryItem(
                        id=doc_id,
                        content=results["documents"][i],
                        memory_type=MemoryType(meta.get("memory_type", "fact")),
                        category=meta.get("category", "general"),
                        confidence=meta.get("confidence", 0.8),
                        usage_count=meta.get("usage_count", 0),
                        created_at=datetime.fromisoformat(meta["created_at"]) if meta.get("created_at") else datetime.now(),
                        updated_at=datetime.fromisoformat(meta["updated_at"]) if meta.get("updated_at") else datetime.now(),
                    )
                    self.search.add(item)
                    self._update_category_count(item.category)
                except Exception as e:
                    print(f"Error loading memory {doc_id}: {e}")

    def _generate_id(self, content: str) -> str:
        """Generate unique ID for memory."""
        return hashlib.md5(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:12]

    def _update_category_count(self, category: str):
        """Update category item count."""
        if category not in self.categories:
            self.categories[category] = MemoryCategory(
                name=category,
                description=f"Memories about {category}"
            )
        self.categories[category].item_count += 1

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        category: str = "general",
        confidence: float = 0.8,
        source: str = "user"
    ) -> Optional[MemoryItem]:
        """Store a new memory."""
        # Check for duplicates
        existing_id = self.search.is_duplicate(content)
        if existing_id:
            return self._reinforce(existing_id)

        # Create new item
        item = MemoryItem(
            id=self._generate_id(content),
            content=content,
            memory_type=memory_type,
            category=category,
            confidence=confidence,
            source=source
        )

        # Add to indices
        self.search.add(item)
        self._update_category_count(category)

        # Add to ChromaDB
        self.collection.add(
            ids=[item.id],
            documents=[item.content],
            metadatas=[{
                "memory_type": item.memory_type.value,
                "category": item.category,
                "confidence": item.confidence,
                "usage_count": item.usage_count,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat(),
            }]
        )

        print(f"Remembered: {content[:50]}...")
        return item

    def _reinforce(self, item_id: str) -> Optional[MemoryItem]:
        """Reinforce an existing memory."""
        idx = self.search.item_index.get(item_id)
        if idx is None:
            return None

        item = self.search.items[idx]
        item.usage_count += 1
        item.confidence = min(1.0, item.confidence + 0.05)
        item.updated_at = datetime.now()

        # Update ChromaDB
        self.collection.update(
            ids=[item.id],
            metadatas=[{
                "memory_type": item.memory_type.value,
                "category": item.category,
                "confidence": item.confidence,
                "usage_count": item.usage_count,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat(),
            }]
        )

        print(f"Reinforced: {item.content[:50]}... (confidence: {item.confidence:.2f})")
        return item

    def recall(
        self,
        query: str,
        n_results: int = 5,
        memory_types: Optional[List[MemoryType]] = None,
        min_confidence: float = 0.3
    ) -> List[Tuple[MemoryItem, float]]:
        """Search memories using multi-index fusion."""
        results = {}

        # BM25 scores (keyword)
        bm25_results = self.search.bm25_search(query, n_results * 2)
        for item_id, score in bm25_results:
            results[item_id] = results.get(item_id, 0) + score * 0.3

        # Vector scores (semantic)
        if self.collection.count() > 0:
            vector_results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results * 2, self.collection.count())
            )
            if vector_results["ids"] and vector_results["ids"][0]:
                for i, item_id in enumerate(vector_results["ids"][0]):
                    distance = vector_results["distances"][0][i] if vector_results["distances"] else 1.0
                    score = 1 - distance  # Convert distance to similarity
                    results[item_id] = results.get(item_id, 0) + score * 0.5

        # Time decay (recent = higher)
        now = datetime.now()
        for item in self.search.items:
            age_days = (now - item.created_at).days
            time_score = 1 / (1 + age_days * 0.1)  # Decay over time
            results[item.id] = results.get(item.id, 0) + time_score * 0.2

        # Filter and sort
        final_results = []
        for item in self.search.items:
            if item.id not in results:
                continue
            if item.confidence < min_confidence:
                continue
            if memory_types and item.memory_type not in memory_types:
                continue

            # Mark as used
            item.usage_count += 1
            item.last_used = datetime.now()

            final_results.append((item, results[item.id]))

        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:n_results]

    def what_happened(self, time_description: str) -> List[MemoryItem]:
        """Temporal query: 'What did I do today?'"""
        now = datetime.now()

        time_desc = time_description.lower()

        if "today" in time_desc:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = now
        elif "yesterday" in time_desc:
            yesterday = now - timedelta(days=1)
            start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            end = yesterday.replace(hour=23, minute=59, second=59)
        elif "this week" in time_desc or "last week" in time_desc:
            start = now - timedelta(weeks=1)
            end = now
        elif "this morning" in time_desc:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = now.replace(hour=12, minute=0, second=0)
        elif "this afternoon" in time_desc:
            start = now.replace(hour=12, minute=0, second=0, microsecond=0)
            end = now.replace(hour=18, minute=0, second=0)
        else:
            # Default to last 24 hours
            start = now - timedelta(days=1)
            end = now

        return self.search.time_search(start, end)

    def get_all_facts(self) -> List[str]:
        """Get all stored facts (backward compatibility)."""
        return [item.content for item in self.search.items]

    def get_context_for_request(self, query: str) -> str:
        """Get memory context for a request (backward compatibility)."""
        results = self.recall(query, n_results=5)

        if not results:
            return ""

        # For small datasets, be less strict about score filtering
        # For larger datasets, filter by score threshold
        lines = ["Things I remember:"]
        if len(self.search.items) <= 5:
            # Small dataset: include top results regardless of score
            for item, score in results[:3]:
                lines.append(f"- {item.content}")
        else:
            # Larger dataset: filter by score
            for item, score in results:
                if score > 0.3:
                    lines.append(f"- {item.content}")

        return "\n".join(lines) if len(lines) > 1 else ""


# Singleton
_memory_v2: Optional[HierarchicalMemory] = None

def get_memory_v2() -> HierarchicalMemory:
    """Get the hierarchical memory instance."""
    global _memory_v2
    if _memory_v2 is None:
        _memory_v2 = HierarchicalMemory()
    return _memory_v2
