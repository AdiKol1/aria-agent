"""
Memory Pruner

Intelligent lifecycle management for memories.
Different memory types have different decay policies.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any, Tuple

from .types import MemoryType, PruningPolicy


class MemoryPruner:
    """
    Manages memory lifecycle with type-specific pruning policies.

    Policies:
    - Preferences: Never auto-decay (user preferences are sacred)
    - Facts: Flag on contradiction (don't overwrite, ask for clarification)
    - Patterns: Archive if unused 30+ days or success rate <30%
    - Insights: Archive after 7 days if not acted upon
    - Skills: Archive if consistently failing
    - Interactions: Remove oldest first when over limit

    Usage:
        pruner = MemoryPruner(memory_system)

        # Run pruning (typically called periodically)
        results = pruner.prune()

        # Check for contradictions before adding a fact
        contradictions = pruner.check_contradictions(new_fact, existing_facts)

        # Get pruning stats
        stats = pruner.get_stats()
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        memory_system: Any = None,
    ):
        """
        Initialize the memory pruner.

        Args:
            storage_path: Where to persist archive
            memory_system: Reference to AriaMemory for integration
        """
        self.storage_path = storage_path or Path.home() / ".aria" / "archive"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.memory_system = memory_system

        # Policies for each memory type
        self.policies: Dict[MemoryType, PruningPolicy] = {
            mt: PruningPolicy.for_type(mt) for mt in MemoryType
        }

        # Archive storage
        self.archive: Dict[str, Dict[str, Any]] = {}

        # Flagged contradictions (fact_id -> [contradicting_fact_ids])
        self.contradictions: Dict[str, List[str]] = {}

        # Callbacks
        self.on_memory_archived: Optional[Callable[[str, MemoryType, str], None]] = None
        self.on_contradiction_found: Optional[Callable[[str, str], None]] = None

        # Load existing archive
        self._load_archive()

    def _load_archive(self) -> None:
        """Load archived memories."""
        archive_file = self.storage_path / "archive.json"
        if archive_file.exists():
            try:
                with open(archive_file) as f:
                    data = json.load(f)
                self.archive = data.get("archive", {})
                self.contradictions = data.get("contradictions", {})
            except Exception as e:
                print(f"Error loading archive: {e}")

    def _save_archive(self) -> None:
        """Persist archive to storage."""
        archive_file = self.storage_path / "archive.json"
        try:
            data = {
                "archive": self.archive,
                "contradictions": self.contradictions,
                "updated_at": datetime.now().isoformat(),
            }
            with open(archive_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving archive: {e}")

    # =========================================================================
    # Pruning
    # =========================================================================

    def prune(
        self,
        memories: Optional[Dict[str, Dict[str, Any]]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Run pruning on all memories.

        Args:
            memories: Dict of memory_id -> memory_data (optional, uses memory_system if not provided)
            dry_run: If True, don't actually archive anything

        Returns:
            Dict with pruning results
        """
        results = {
            "archived": [],
            "flagged": [],
            "skipped": [],
            "by_type": {},
        }

        if memories is None and self.memory_system is None:
            return results

        # Get memories from the system if not provided
        if memories is None:
            memories = self._get_all_memories()

        now = datetime.now()

        for memory_id, memory_data in memories.items():
            memory_type = self._infer_memory_type(memory_data)
            policy = self.policies.get(memory_type, PruningPolicy(memory_type=memory_type))

            # Initialize type counter
            if memory_type.value not in results["by_type"]:
                results["by_type"][memory_type.value] = {"checked": 0, "archived": 0}
            results["by_type"][memory_type.value]["checked"] += 1

            # Check if should prune
            should_archive, reason = self._should_prune(memory_data, policy, now)

            if should_archive:
                if not dry_run:
                    self._archive_memory(memory_id, memory_data, reason)
                results["archived"].append({
                    "id": memory_id,
                    "type": memory_type.value,
                    "reason": reason,
                })
                results["by_type"][memory_type.value]["archived"] += 1
            else:
                results["skipped"].append(memory_id)

        if not dry_run:
            self._save_archive()

        return results

    def _should_prune(
        self,
        memory: Dict[str, Any],
        policy: PruningPolicy,
        now: datetime,
    ) -> Tuple[bool, str]:
        """
        Determine if a memory should be pruned.

        Returns (should_prune, reason)
        """
        # Never auto-delete protected types
        if policy.never_auto_delete:
            return False, "protected"

        # Check auto-decay
        if policy.auto_decay:
            last_used = memory.get("last_used") or memory.get("updated_at") or memory.get("created_at")
            if last_used:
                if isinstance(last_used, str):
                    last_used = datetime.fromisoformat(last_used)
                days_inactive = (now - last_used).days
                if days_inactive > policy.decay_days:
                    return True, f"inactive for {days_inactive} days"

        # Check performance threshold
        if policy.min_success_rate > 0:
            success_rate = memory.get("success_rate", 1.0)
            if success_rate < policy.min_success_rate:
                return True, f"success rate {success_rate:.0%} below threshold"

        # Check minimum usage
        if policy.min_usage_count > 0:
            usage_count = memory.get("usage_count") or memory.get("times_applied", 0)
            times_executed = memory.get("times_executed", 0)
            total_usage = usage_count + times_executed

            # Only prune if used enough times but still failing
            if total_usage >= policy.min_usage_count:
                success_rate = memory.get("success_rate", 1.0)
                if success_rate < policy.min_success_rate:
                    return True, f"poor performance after {total_usage} uses"

        return False, ""

    def _archive_memory(
        self,
        memory_id: str,
        memory_data: Dict[str, Any],
        reason: str,
    ) -> None:
        """
        Archive a memory (soft delete).

        Args:
            memory_id: ID of the memory
            memory_data: The memory data
            reason: Why it's being archived
        """
        self.archive[memory_id] = {
            **memory_data,
            "archived_at": datetime.now().isoformat(),
            "archive_reason": reason,
        }

        memory_type = self._infer_memory_type(memory_data)
        if self.on_memory_archived:
            self.on_memory_archived(memory_id, memory_type, reason)

        print(f"Archived {memory_type.value}: {memory_id} ({reason})")

    def _infer_memory_type(self, memory: Dict[str, Any]) -> MemoryType:
        """Infer the type of memory from its data."""
        category = memory.get("category", "").lower()

        if category in ["preference", "preferences"]:
            return MemoryType.PREFERENCE
        elif category in ["fact", "personal", "work"]:
            return MemoryType.FACT
        elif category == "pattern":
            return MemoryType.PATTERN
        elif category == "insight":
            return MemoryType.INSIGHT
        elif category == "skill" or "actions" in memory:
            return MemoryType.SKILL
        elif category == "interaction" or "user_request" in memory:
            return MemoryType.INTERACTION
        else:
            return MemoryType.FACT  # Default

    def _get_all_memories(self) -> Dict[str, Dict[str, Any]]:
        """Get all memories from the memory system."""
        if not self.memory_system:
            return {}

        memories = {}

        # Get facts
        try:
            if hasattr(self.memory_system, 'facts'):
                results = self.memory_system.facts.get()
                if results and results.get('ids'):
                    for i, id_ in enumerate(results['ids']):
                        memories[id_] = {
                            "id": id_,
                            "content": results['documents'][i] if results.get('documents') else "",
                            "category": results['metadatas'][i].get('category', 'fact') if results.get('metadatas') else 'fact',
                            **(results['metadatas'][i] if results.get('metadatas') else {}),
                        }
        except Exception as e:
            print(f"Error getting facts: {e}")

        return memories

    # =========================================================================
    # Contradiction Detection
    # =========================================================================

    def check_contradictions(
        self,
        new_fact: str,
        existing_facts: List[Dict[str, Any]],
        similarity_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Check if a new fact contradicts existing facts.

        Uses simple heuristics to detect potential contradictions:
        - Same subject but different value
        - Opposite statements

        Args:
            new_fact: The new fact to check
            existing_facts: List of existing facts to check against
            similarity_threshold: How similar facts need to be to be considered related

        Returns:
            List of potentially contradicting facts
        """
        contradictions = []
        new_fact_lower = new_fact.lower()

        # Extract key patterns from new fact
        negation_words = ["not", "don't", "doesn't", "never", "no longer", "stopped"]
        preference_patterns = ["likes", "prefers", "wants", "uses", "favorite"]

        for existing in existing_facts:
            existing_text = existing.get("content", existing.get("fact", "")).lower()

            # Skip if too different
            if not self._are_related(new_fact_lower, existing_text):
                continue

            # Check for direct contradiction patterns
            is_contradiction = False
            reason = ""

            # Pattern 1: Opposite preferences
            # "likes dark mode" vs "likes light mode"
            for pattern in preference_patterns:
                if pattern in new_fact_lower and pattern in existing_text:
                    # Same preference type, check if value differs
                    new_value = self._extract_value_after(new_fact_lower, pattern)
                    existing_value = self._extract_value_after(existing_text, pattern)
                    if new_value and existing_value and new_value != existing_value:
                        is_contradiction = True
                        reason = f"Different values for same preference: '{new_value}' vs '{existing_value}'"

            # Pattern 2: Negation
            # "likes coffee" vs "doesn't like coffee"
            for neg in negation_words:
                if neg in new_fact_lower and neg not in existing_text:
                    # New fact has negation, old doesn't
                    is_contradiction = True
                    reason = f"New fact negates existing (contains '{neg}')"
                elif neg in existing_text and neg not in new_fact_lower:
                    # Old fact has negation, new doesn't
                    is_contradiction = True
                    reason = f"New fact contradicts negation in existing"

            if is_contradiction:
                contradictions.append({
                    **existing,
                    "contradiction_reason": reason,
                })

        return contradictions

    def _are_related(self, text1: str, text2: str) -> bool:
        """Check if two facts are about the same subject."""
        # Simple word overlap check
        words1 = set(text1.split())
        words2 = set(text2.split())

        # Remove common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or", "in", "on", "at"}
        words1 = words1 - stopwords
        words2 = words2 - stopwords

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        min_len = min(len(words1), len(words2))

        return overlap / min_len > 0.3

    def _extract_value_after(self, text: str, keyword: str) -> Optional[str]:
        """Extract the value that comes after a keyword."""
        try:
            idx = text.index(keyword)
            after = text[idx + len(keyword):].strip()
            # Get first few words as the value
            words = after.split()[:3]
            return " ".join(words)
        except ValueError:
            return None

    def flag_contradiction(
        self,
        fact_id: str,
        contradicting_id: str,
    ) -> None:
        """
        Flag two facts as potentially contradicting.

        Instead of auto-deleting, we flag for user review.

        Args:
            fact_id: The primary fact
            contradicting_id: The contradicting fact
        """
        if fact_id not in self.contradictions:
            self.contradictions[fact_id] = []

        if contradicting_id not in self.contradictions[fact_id]:
            self.contradictions[fact_id].append(contradicting_id)

        if self.on_contradiction_found:
            self.on_contradiction_found(fact_id, contradicting_id)

        self._save_archive()
        print(f"Flagged contradiction between {fact_id} and {contradicting_id}")

    def get_flagged_contradictions(self) -> Dict[str, List[str]]:
        """Get all flagged contradictions for review."""
        return self.contradictions.copy()

    def resolve_contradiction(
        self,
        fact_id: str,
        keep: bool = True,
        archive_contradicting: bool = True,
    ) -> None:
        """
        Resolve a flagged contradiction.

        Args:
            fact_id: The fact to resolve
            keep: If True, keep this fact. If False, archive it.
            archive_contradicting: If True, archive the contradicting facts.
        """
        if fact_id not in self.contradictions:
            return

        contradicting_ids = self.contradictions[fact_id]

        if not keep:
            # Archive the main fact
            # Would need memory_system integration to actually archive
            pass

        if archive_contradicting:
            # Archive contradicting facts
            # Would need memory_system integration
            pass

        # Clear the flag
        del self.contradictions[fact_id]
        self._save_archive()

    # =========================================================================
    # Archive Management
    # =========================================================================

    def restore_from_archive(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Restore a memory from archive.

        Args:
            memory_id: ID of the memory to restore

        Returns:
            The restored memory data, or None if not found
        """
        if memory_id not in self.archive:
            return None

        memory_data = self.archive.pop(memory_id)

        # Remove archive metadata
        memory_data.pop("archived_at", None)
        memory_data.pop("archive_reason", None)

        self._save_archive()
        print(f"Restored memory: {memory_id}")
        return memory_data

    def list_archived(
        self,
        memory_type: Optional[MemoryType] = None,
    ) -> List[Dict[str, Any]]:
        """
        List archived memories.

        Args:
            memory_type: Filter by type (optional)

        Returns:
            List of archived memories
        """
        result = []
        for id_, data in self.archive.items():
            if memory_type:
                inferred_type = self._infer_memory_type(data)
                if inferred_type != memory_type:
                    continue
            result.append({"id": id_, **data})
        return result

    def permanently_delete(self, memory_id: str) -> bool:
        """
        Permanently delete a memory from archive.

        This is irreversible!

        Args:
            memory_id: ID of the memory to delete

        Returns:
            True if deleted, False if not found
        """
        if memory_id in self.archive:
            del self.archive[memory_id]
            self._save_archive()
            return True
        return False

    def clear_archive(
        self,
        older_than_days: Optional[int] = None,
    ) -> int:
        """
        Clear archived memories.

        Args:
            older_than_days: Only clear items archived more than this many days ago

        Returns:
            Number of items cleared
        """
        if older_than_days is None:
            count = len(self.archive)
            self.archive = {}
            self._save_archive()
            return count

        cutoff = datetime.now() - timedelta(days=older_than_days)
        to_delete = []

        for id_, data in self.archive.items():
            archived_at = data.get("archived_at")
            if archived_at:
                archived_date = datetime.fromisoformat(archived_at)
                if archived_date < cutoff:
                    to_delete.append(id_)

        for id_ in to_delete:
            del self.archive[id_]

        self._save_archive()
        return len(to_delete)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get pruning statistics."""
        archive_by_type = {}
        archive_by_reason = {}

        for data in self.archive.values():
            # By type
            memory_type = self._infer_memory_type(data)
            type_name = memory_type.value
            archive_by_type[type_name] = archive_by_type.get(type_name, 0) + 1

            # By reason
            reason = data.get("archive_reason", "unknown")
            archive_by_reason[reason] = archive_by_reason.get(reason, 0) + 1

        return {
            "total_archived": len(self.archive),
            "by_type": archive_by_type,
            "by_reason": archive_by_reason,
            "flagged_contradictions": len(self.contradictions),
            "policies": {
                mt.value: {
                    "auto_decay": self.policies[mt].auto_decay,
                    "decay_days": self.policies[mt].decay_days,
                    "min_success_rate": self.policies[mt].min_success_rate,
                    "never_auto_delete": self.policies[mt].never_auto_delete,
                }
                for mt in MemoryType
            },
        }

    def update_policy(
        self,
        memory_type: MemoryType,
        **kwargs,
    ) -> None:
        """
        Update a pruning policy.

        Args:
            memory_type: The memory type to update
            **kwargs: Policy attributes to update
        """
        policy = self.policies.get(memory_type)
        if policy:
            for key, value in kwargs.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)


# Singleton instance
_pruner: Optional[MemoryPruner] = None


def get_memory_pruner() -> MemoryPruner:
    """Get the singleton MemoryPruner instance."""
    global _pruner
    if _pruner is None:
        _pruner = MemoryPruner()
    return _pruner
