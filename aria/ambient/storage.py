"""
Aria Ambient Intelligence - Storage Layer

Provides persistent storage for worlds, signals, and insights.
Uses YAML for world configuration and JSON for signal/insight history.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import threading
import shutil

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from .models import World, Signal, Insight, ConsciousnessState, now_iso
from .constants import STORAGE_PATHS

logger = logging.getLogger(__name__)


def get_storage_path(name: str) -> Path:
    """Get the storage path for a given storage type."""
    base = Path.home() / ".aria"
    subpath = STORAGE_PATHS.get(name, f"ambient/{name}")
    return base / subpath


class WorldStorage:
    """
    YAML-based storage for world configurations.

    Each world is stored as a separate YAML file for easy editing
    and version control.
    """

    def __init__(self, storage_path: Path = None):
        """
        Initialize world storage.

        Args:
            storage_path: Directory for world files. Defaults to ~/.aria/ambient/worlds
        """
        self._path = storage_path or get_storage_path("worlds")
        self._path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        if not HAS_YAML:
            logger.warning("PyYAML not installed, using JSON format for worlds")

    def _world_file(self, world_id: str) -> Path:
        """Get the file path for a world."""
        ext = "yaml" if HAS_YAML else "json"
        return self._path / f"{world_id}.{ext}"

    def save_world(self, world: World) -> bool:
        """
        Save a world to storage.

        Uses atomic write (write to temp file, then rename) for safety.

        Args:
            world: World object to save

        Returns:
            True if successful
        """
        try:
            with self._lock:
                file_path = self._world_file(world.id)
                temp_path = file_path.with_suffix(".tmp")

                data = world.to_dict()

                if HAS_YAML:
                    with open(temp_path, "w") as f:
                        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
                else:
                    with open(temp_path, "w") as f:
                        json.dump(data, f, indent=2)

                # Atomic rename
                temp_path.rename(file_path)

                logger.debug(f"Saved world: {world.name}")
                return True

        except Exception as e:
            logger.error(f"Failed to save world {world.id}: {e}")
            return False

    def load_world(self, world_id: str) -> Optional[World]:
        """
        Load a world from storage.

        Args:
            world_id: ID of the world to load

        Returns:
            World object, or None if not found
        """
        try:
            with self._lock:
                # Try YAML first, then JSON
                yaml_path = self._path / f"{world_id}.yaml"
                json_path = self._path / f"{world_id}.json"

                if yaml_path.exists() and HAS_YAML:
                    with open(yaml_path, "r") as f:
                        data = yaml.safe_load(f)
                elif json_path.exists():
                    with open(json_path, "r") as f:
                        data = json.load(f)
                else:
                    return None

                return World.from_dict(data)

        except Exception as e:
            logger.error(f"Failed to load world {world_id}: {e}")
            return None

    def delete_world(self, world_id: str) -> bool:
        """
        Delete a world from storage.

        Creates a backup before deletion.

        Args:
            world_id: ID of the world to delete

        Returns:
            True if successful
        """
        try:
            with self._lock:
                yaml_path = self._path / f"{world_id}.yaml"
                json_path = self._path / f"{world_id}.json"

                for file_path in [yaml_path, json_path]:
                    if file_path.exists():
                        # Backup
                        backup_path = file_path.with_suffix(f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                        shutil.copy(file_path, backup_path)

                        # Delete
                        file_path.unlink()
                        logger.info(f"Deleted world file: {file_path}")
                        return True

                return False

        except Exception as e:
            logger.error(f"Failed to delete world {world_id}: {e}")
            return False

    def list_world_ids(self) -> List[str]:
        """
        List all world IDs in storage.

        Returns:
            List of world IDs
        """
        try:
            world_ids = set()
            for pattern in ["*.yaml", "*.json"]:
                for file_path in self._path.glob(pattern):
                    if not file_path.name.startswith(".") and "backup" not in file_path.name:
                        world_ids.add(file_path.stem)
            return sorted(world_ids)

        except Exception as e:
            logger.error(f"Failed to list worlds: {e}")
            return []

    def backup_all(self, backup_dir: Path = None) -> bool:
        """
        Create a backup of all worlds.

        Args:
            backup_dir: Backup directory. Defaults to storage_path/backups

        Returns:
            True if successful
        """
        try:
            backup_dir = backup_dir or self._path / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"worlds_{timestamp}"
            backup_path.mkdir()

            for pattern in ["*.yaml", "*.json"]:
                for file_path in self._path.glob(pattern):
                    if "backup" not in file_path.name:
                        shutil.copy(file_path, backup_path / file_path.name)

            logger.info(f"Backed up worlds to: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to backup worlds: {e}")
            return False


class SignalCache:
    """
    In-memory cache for signals with automatic expiration.

    Signals are stored temporarily before processing by the cortex.
    Old signals are automatically cleaned up.
    """

    def __init__(self, max_age_seconds: int = 3600):
        """
        Initialize signal cache.

        Args:
            max_age_seconds: Maximum age of signals in seconds (default 1 hour)
        """
        self._max_age = max_age_seconds
        self._signals: Dict[str, Signal] = {}
        self._lock = threading.Lock()

    def add_signal(self, signal: Signal) -> None:
        """Add a signal to the cache."""
        with self._lock:
            self._signals[signal.id] = signal

    def add_signals(self, signals: List[Signal]) -> int:
        """
        Add multiple signals to the cache.

        Returns:
            Number of signals added
        """
        with self._lock:
            for signal in signals:
                self._signals[signal.id] = signal
            return len(signals)

    def get_signal(self, signal_id: str) -> Optional[Signal]:
        """Get a signal by ID."""
        with self._lock:
            return self._signals.get(signal_id)

    def get_signals(
        self,
        since: datetime = None,
        signal_type: str = None,
        source: str = None,
    ) -> List[Signal]:
        """
        Get signals with optional filtering.

        Args:
            since: Only signals after this time
            signal_type: Filter by signal type
            source: Filter by source watcher

        Returns:
            List of matching signals
        """
        with self._lock:
            results = list(self._signals.values())

        # Apply filters
        if since:
            results = [
                s for s in results
                if datetime.fromisoformat(s.timestamp) > since
            ]

        if signal_type:
            results = [s for s in results if s.type.value == signal_type]

        if source:
            results = [s for s in results if s.source == source]

        # Sort by timestamp (newest first)
        results.sort(key=lambda s: s.timestamp, reverse=True)
        return results

    def get_unprocessed(self) -> List[Signal]:
        """Get all signals that haven't been processed yet."""
        with self._lock:
            return [
                s for s in self._signals.values()
                if not s.relevant_worlds  # Not yet scored
            ]

    def clear_old(self) -> int:
        """
        Remove signals older than max_age.

        Returns:
            Number of signals removed
        """
        with self._lock:
            cutoff = datetime.now() - timedelta(seconds=self._max_age)
            old_ids = [
                sid for sid, signal in self._signals.items()
                if datetime.fromisoformat(signal.timestamp) < cutoff
            ]

            for sid in old_ids:
                del self._signals[sid]

            if old_ids:
                logger.debug(f"Cleared {len(old_ids)} old signals from cache")

            return len(old_ids)

    def clear_expired(self) -> int:
        """
        Remove signals that have explicitly expired.

        Returns:
            Number of signals removed
        """
        with self._lock:
            now = datetime.now()
            expired_ids = [
                sid for sid, signal in self._signals.items()
                if signal.expires_at and datetime.fromisoformat(signal.expires_at) < now
            ]

            for sid in expired_ids:
                del self._signals[sid]

            return len(expired_ids)

    def clear_all(self) -> int:
        """Clear all signals from cache."""
        with self._lock:
            count = len(self._signals)
            self._signals.clear()
            return count

    def count(self) -> int:
        """Get number of signals in cache."""
        with self._lock:
            return len(self._signals)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            by_source = {}
            by_type = {}

            for signal in self._signals.values():
                by_source[signal.source] = by_source.get(signal.source, 0) + 1
                by_type[signal.type.value] = by_type.get(signal.type.value, 0) + 1

            return {
                "total_signals": len(self._signals),
                "max_age_seconds": self._max_age,
                "by_source": by_source,
                "by_type": by_type,
            }


class InsightHistory:
    """
    Persistent storage for insights and their outcomes.

    Stores insights in JSON format with outcome tracking for learning.
    """

    def __init__(self, storage_path: Path = None):
        """
        Initialize insight history.

        Args:
            storage_path: Directory for insight files. Defaults to ~/.aria/ambient/insights
        """
        self._path = storage_path or get_storage_path("insights")
        self._path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _get_file_for_date(self, date: datetime) -> Path:
        """Get the file path for a specific date."""
        return self._path / f"insights_{date.strftime('%Y_%m_%d')}.json"

    def record_insight(self, insight: Insight) -> bool:
        """
        Record a new insight.

        Args:
            insight: Insight to record

        Returns:
            True if successful
        """
        try:
            with self._lock:
                today = datetime.now()
                file_path = self._get_file_for_date(today)

                # Load existing data
                data = []
                if file_path.exists():
                    with open(file_path, "r") as f:
                        data = json.load(f)

                # Add new insight
                data.append({
                    **insight.to_dict(),
                    "outcome": None,
                    "outcome_recorded_at": None,
                })

                # Save
                with open(file_path, "w") as f:
                    json.dump(data, f, indent=2)

                logger.debug(f"Recorded insight: {insight.title}")
                return True

        except Exception as e:
            logger.error(f"Failed to record insight: {e}")
            return False

    def record_outcome(
        self,
        insight_id: str,
        outcome: str,
        notes: str = ""
    ) -> bool:
        """
        Record the outcome of an insight.

        Args:
            insight_id: ID of the insight
            outcome: Outcome value (acted, dismissed, expired, helpful, not_helpful)
            notes: Optional notes about the outcome

        Returns:
            True if successful
        """
        try:
            with self._lock:
                # Search recent files for the insight
                for days_back in range(30):
                    date = datetime.now() - timedelta(days=days_back)
                    file_path = self._get_file_for_date(date)

                    if not file_path.exists():
                        continue

                    with open(file_path, "r") as f:
                        data = json.load(f)

                    # Find and update the insight
                    for item in data:
                        if item.get("id") == insight_id:
                            item["outcome"] = outcome
                            item["outcome_notes"] = notes
                            item["outcome_recorded_at"] = now_iso()

                            with open(file_path, "w") as f:
                                json.dump(data, f, indent=2)

                            logger.debug(f"Recorded outcome for insight {insight_id}: {outcome}")
                            return True

                logger.warning(f"Insight not found: {insight_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to record outcome: {e}")
            return False

    def get_recent(self, days: int = 7) -> List[Insight]:
        """
        Get insights from the last N days.

        Args:
            days: Number of days to look back

        Returns:
            List of Insight objects
        """
        insights = []

        try:
            with self._lock:
                for days_back in range(days):
                    date = datetime.now() - timedelta(days=days_back)
                    file_path = self._get_file_for_date(date)

                    if file_path.exists():
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            for item in data:
                                insights.append(Insight.from_dict(item))

        except Exception as e:
            logger.error(f"Failed to get recent insights: {e}")

        # Sort by creation time (newest first)
        insights.sort(key=lambda i: i.created_at, reverse=True)
        return insights

    def get_by_world(self, world_id: str, days: int = 30) -> List[Insight]:
        """
        Get insights for a specific world.

        Args:
            world_id: World ID to filter by
            days: Number of days to look back

        Returns:
            List of Insight objects for the world
        """
        all_insights = self.get_recent(days)
        return [i for i in all_insights if i.world_id == world_id]

    def get_outcomes_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get summary of insight outcomes for learning.

        Returns:
            Dictionary with outcome statistics
        """
        insights = self.get_recent(days)

        outcomes = {}
        by_world = {}
        by_priority = {}

        for insight in insights:
            insight_data = insight.to_dict()
            outcome = insight_data.get("outcome")

            if outcome:
                outcomes[outcome] = outcomes.get(outcome, 0) + 1

            # Track by world
            world_id = insight.world_id
            if world_id not in by_world:
                by_world[world_id] = {"total": 0, "acted": 0}
            by_world[world_id]["total"] += 1
            if outcome == "acted":
                by_world[world_id]["acted"] += 1

            # Track by priority
            priority = insight.priority.value
            if priority not in by_priority:
                by_priority[priority] = {"total": 0, "acted": 0}
            by_priority[priority]["total"] += 1
            if outcome == "acted":
                by_priority[priority]["acted"] += 1

        return {
            "total_insights": len(insights),
            "outcomes": outcomes,
            "by_world": by_world,
            "by_priority": by_priority,
        }

    def cleanup_old(self, days_to_keep: int = 90) -> int:
        """
        Remove insight files older than the specified number of days.

        Args:
            days_to_keep: Number of days of data to keep

        Returns:
            Number of files removed
        """
        try:
            with self._lock:
                cutoff = datetime.now() - timedelta(days=days_to_keep)
                removed = 0

                for file_path in self._path.glob("insights_*.json"):
                    # Parse date from filename
                    try:
                        date_str = file_path.stem.replace("insights_", "")
                        file_date = datetime.strptime(date_str, "%Y_%m_%d")

                        if file_date < cutoff:
                            file_path.unlink()
                            removed += 1
                            logger.debug(f"Removed old insight file: {file_path.name}")

                    except ValueError:
                        continue  # Skip files with unexpected names

                if removed:
                    logger.info(f"Cleaned up {removed} old insight files")
                return removed

        except Exception as e:
            logger.error(f"Failed to cleanup old insights: {e}")
            return 0


class ConsciousnessStorage:
    """
    Storage for Aria's consciousness state.

    Persists the current mental state between sessions.
    """

    def __init__(self, storage_path: Path = None):
        """
        Initialize consciousness storage.

        Args:
            storage_path: File for consciousness state. Defaults to ~/.aria/ambient/consciousness/state.json
        """
        base_path = storage_path or get_storage_path("consciousness")
        base_path.mkdir(parents=True, exist_ok=True)
        self._file_path = base_path / "state.json"
        self._lock = threading.Lock()

    def save_state(self, state: ConsciousnessState) -> bool:
        """Save consciousness state."""
        try:
            with self._lock:
                data = state.to_dict()
                with open(self._file_path, "w") as f:
                    json.dump(data, f, indent=2)
                return True

        except Exception as e:
            logger.error(f"Failed to save consciousness state: {e}")
            return False

    def load_state(self) -> Optional[ConsciousnessState]:
        """Load consciousness state."""
        try:
            with self._lock:
                if not self._file_path.exists():
                    return None

                with open(self._file_path, "r") as f:
                    data = json.load(f)
                return ConsciousnessState.from_dict(data)

        except Exception as e:
            logger.error(f"Failed to load consciousness state: {e}")
            return None

    def clear_state(self) -> bool:
        """Clear the consciousness state."""
        try:
            with self._lock:
                if self._file_path.exists():
                    self._file_path.unlink()
                return True

        except Exception as e:
            logger.error(f"Failed to clear consciousness state: {e}")
            return False
