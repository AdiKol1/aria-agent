"""
Pattern Learner

Learns patterns from user corrections and repeated behaviors.
Promotes observations to patterns when thresholds are met.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

from .types import (
    ObservationType,
    Observation,
    LearnedPattern,
)


class PatternLearner:
    """
    Learns patterns from user behavior without explicit teaching.

    Observes:
    - Corrections: When user fixes Aria's mistakes
    - Repeated actions: When user does the same thing multiple times
    - Consistent choices: When user always picks the same option
    - Failure recovery: When user fixes something Aria broke

    Promotes observations to patterns when enough evidence accumulates.

    Usage:
        learner = PatternLearner()

        # Record a correction
        learner.observe_correction(
            original="formatted as plain text",
            corrected="formatted as code block",
            context={"task": "sharing_code"}
        )

        # Record repeated action
        learner.observe_repeated_action(
            action="save file",
            context={"app": "VS Code"}
        )

        # Check for applicable patterns
        patterns = learner.get_patterns_for_context({"task": "sharing_code"})
    """

    # Thresholds for promoting observations to patterns
    CORRECTION_THRESHOLD = 2  # 2 corrections on same thing → pattern
    REPEATED_ACTION_THRESHOLD = 3  # 3 repeated actions → pattern
    CONSISTENT_CHOICE_THRESHOLD = 3  # 3 consistent choices → pattern
    FAILURE_RECOVERY_THRESHOLD = 2  # 2 recoveries → pattern

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the pattern learner.

        Args:
            storage_path: Where to persist patterns and observations
        """
        self.storage_path = storage_path or Path.home() / ".aria" / "patterns"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.observations: Dict[str, Observation] = {}
        self.patterns: Dict[str, LearnedPattern] = {}

        # Callbacks
        self.on_pattern_learned: Optional[Callable[[LearnedPattern], None]] = None
        self.on_observation_recorded: Optional[Callable[[Observation], None]] = None

        # Load existing data
        self._load_data()

    def _load_data(self) -> None:
        """Load observations and patterns from storage."""
        # Load observations
        obs_file = self.storage_path / "observations.json"
        if obs_file.exists():
            try:
                with open(obs_file) as f:
                    data = json.load(f)
                for obs_data in data.get("observations", []):
                    obs = Observation.from_dict(obs_data)
                    self.observations[obs.id] = obs
            except Exception as e:
                print(f"Error loading observations: {e}")

        # Load patterns
        patterns_file = self.storage_path / "patterns.json"
        if patterns_file.exists():
            try:
                with open(patterns_file) as f:
                    data = json.load(f)
                for pattern_data in data.get("patterns", []):
                    pattern = LearnedPattern.from_dict(pattern_data)
                    self.patterns[pattern.id] = pattern
            except Exception as e:
                print(f"Error loading patterns: {e}")

        print(f"Loaded {len(self.observations)} observations, {len(self.patterns)} patterns")

    def _save_data(self) -> None:
        """Persist observations and patterns to storage."""
        # Save observations
        obs_file = self.storage_path / "observations.json"
        try:
            data = {
                "observations": [o.to_dict() for o in self.observations.values()],
                "updated_at": datetime.now().isoformat(),
            }
            with open(obs_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving observations: {e}")

        # Save patterns
        patterns_file = self.storage_path / "patterns.json"
        try:
            data = {
                "patterns": [p.to_dict() for p in self.patterns.values()],
                "updated_at": datetime.now().isoformat(),
            }
            with open(patterns_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving patterns: {e}")

    # =========================================================================
    # Recording Observations
    # =========================================================================

    def _record_observation(self, observation: Observation) -> None:
        """Record an observation and check for pattern promotion."""
        self.observations[observation.id] = observation

        if self.on_observation_recorded:
            self.on_observation_recorded(observation)

        # Find similar observations
        similar = self._find_similar_observations(observation)
        observation.similar_to = [o.id for o in similar]

        # Check if we should promote to pattern
        self._check_pattern_promotion(observation, similar)

        self._save_data()

    def observe_correction(
        self,
        original: str,
        corrected: str,
        context: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> Observation:
        """
        Record when user corrects Aria's action.

        Args:
            original: What Aria did
            corrected: What user changed it to
            context: Context where this happened (app, task, etc.)
            description: Optional description of the correction

        Returns:
            The recorded observation
        """
        obs = Observation(
            id=str(uuid.uuid4())[:8],
            observation_type=ObservationType.CORRECTION,
            description=description or f"Changed '{original}' to '{corrected}'",
            original_action=original,
            corrected_action=corrected,
            context=context or {},
        )
        self._record_observation(obs)
        print(f"Observed correction: {obs.description}")
        return obs

    def observe_repeated_action(
        self,
        action: str,
        context: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> Observation:
        """
        Record when user performs the same action repeatedly.

        Args:
            action: What the user did
            context: Context where this happened
            description: Optional description

        Returns:
            The recorded observation
        """
        obs = Observation(
            id=str(uuid.uuid4())[:8],
            observation_type=ObservationType.REPEATED_ACTION,
            description=description or f"Repeated action: {action}",
            corrected_action=action,  # Use corrected_action to store the action
            context=context or {},
        )
        self._record_observation(obs)
        return obs

    def observe_consistent_choice(
        self,
        choice: str,
        alternatives: List[str],
        context: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> Observation:
        """
        Record when user consistently picks one option over others.

        Args:
            choice: What the user chose
            alternatives: What else was available
            context: Context where this happened
            description: Optional description

        Returns:
            The recorded observation
        """
        obs = Observation(
            id=str(uuid.uuid4())[:8],
            observation_type=ObservationType.CONSISTENT_CHOICE,
            description=description or f"Chose '{choice}' over {alternatives}",
            corrected_action=choice,
            context={
                **(context or {}),
                "alternatives": alternatives,
            },
        )
        self._record_observation(obs)
        return obs

    def observe_failure_recovery(
        self,
        failure: str,
        recovery: str,
        context: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> Observation:
        """
        Record when user fixes something Aria broke.

        Args:
            failure: What went wrong
            recovery: How user fixed it
            context: Context where this happened
            description: Optional description

        Returns:
            The recorded observation
        """
        obs = Observation(
            id=str(uuid.uuid4())[:8],
            observation_type=ObservationType.FAILURE_RECOVERY,
            description=description or f"Recovered from '{failure}' by '{recovery}'",
            original_action=failure,
            corrected_action=recovery,
            context=context or {},
        )
        self._record_observation(obs)
        return obs

    # =========================================================================
    # Pattern Detection and Promotion
    # =========================================================================

    def _find_similar_observations(
        self,
        observation: Observation,
    ) -> List[Observation]:
        """
        Find observations similar to the given one.

        Similarity is based on:
        - Same observation type
        - Similar context
        - Similar actions involved
        """
        similar = []

        for obs in self.observations.values():
            if obs.id == observation.id:
                continue

            # Must be same type
            if obs.observation_type != observation.observation_type:
                continue

            # Check for similar content
            similarity_score = self._calculate_similarity(observation, obs)
            if similarity_score > 0.6:
                similar.append(obs)

        return similar

    def _calculate_similarity(
        self,
        obs1: Observation,
        obs2: Observation,
    ) -> float:
        """
        Calculate similarity between two observations.

        Returns a score from 0.0 to 1.0.
        """
        score = 0.0
        factors = 0

        # Same type gives base score
        if obs1.observation_type == obs2.observation_type:
            score += 0.3
            factors += 1

        # Similar corrected action
        if obs1.corrected_action and obs2.corrected_action:
            if obs1.corrected_action.lower() == obs2.corrected_action.lower():
                score += 0.4
            elif obs1.corrected_action.lower() in obs2.corrected_action.lower() or \
                 obs2.corrected_action.lower() in obs1.corrected_action.lower():
                score += 0.2
            factors += 1

        # Similar original action (for corrections)
        if obs1.original_action and obs2.original_action:
            if obs1.original_action.lower() == obs2.original_action.lower():
                score += 0.3
            factors += 1

        # Similar context
        if obs1.context and obs2.context:
            context_match = 0
            context_total = 0
            for key in set(obs1.context.keys()) | set(obs2.context.keys()):
                if key in obs1.context and key in obs2.context:
                    if obs1.context[key] == obs2.context[key]:
                        context_match += 1
                context_total += 1
            if context_total > 0:
                score += 0.3 * (context_match / context_total)
                factors += 1

        return score / max(factors, 1)

    def _check_pattern_promotion(
        self,
        observation: Observation,
        similar: List[Observation],
    ) -> Optional[LearnedPattern]:
        """
        Check if observations should be promoted to a pattern.

        Returns the new pattern if one was created.
        """
        threshold = {
            ObservationType.CORRECTION: self.CORRECTION_THRESHOLD,
            ObservationType.REPEATED_ACTION: self.REPEATED_ACTION_THRESHOLD,
            ObservationType.CONSISTENT_CHOICE: self.CONSISTENT_CHOICE_THRESHOLD,
            ObservationType.FAILURE_RECOVERY: self.FAILURE_RECOVERY_THRESHOLD,
        }.get(observation.observation_type, 3)

        # Include the current observation
        all_similar = similar + [observation]

        if len(all_similar) >= threshold:
            # Check if a similar pattern already exists
            existing = self._find_existing_pattern(observation)
            if existing:
                # Update existing pattern
                existing.observation_count = len(all_similar)
                existing.evidence = [o.id for o in all_similar]
                existing.confidence = min(1.0, existing.confidence + 0.1)
                print(f"Updated pattern: {existing.trigger}")
                return existing

            # Create new pattern
            pattern = self._create_pattern_from_observations(all_similar)
            if pattern:
                self.patterns[pattern.id] = pattern

                if self.on_pattern_learned:
                    self.on_pattern_learned(pattern)

                print(f"Learned new pattern: {pattern.trigger} → {pattern.action}")
                return pattern

        return None

    def _find_existing_pattern(
        self,
        observation: Observation,
    ) -> Optional[LearnedPattern]:
        """Find an existing pattern that matches this observation."""
        for pattern in self.patterns.values():
            if pattern.is_archived:
                continue

            # Check if contexts match
            context_match = all(
                observation.context.get(k) == v
                for k, v in pattern.context.items()
            )

            # Check if action matches
            action_match = (
                observation.corrected_action and
                observation.corrected_action.lower() in pattern.action.lower()
            )

            if context_match and action_match:
                return pattern

        return None

    def _create_pattern_from_observations(
        self,
        observations: List[Observation],
    ) -> Optional[LearnedPattern]:
        """
        Create a pattern from a list of similar observations.
        """
        if not observations:
            return None

        # Use the most recent observation as the primary
        primary = observations[-1]

        # Merge context from all observations
        merged_context = {}
        for obs in observations:
            for key, value in obs.context.items():
                if key not in merged_context:
                    merged_context[key] = value
                elif merged_context[key] != value:
                    # Context varies, don't include it
                    merged_context.pop(key, None)

        # Generate trigger and action based on observation type
        trigger = ""
        action = ""

        if primary.observation_type == ObservationType.CORRECTION:
            trigger = f"When doing something that would result in '{primary.original_action}'"
            action = f"Instead, {primary.corrected_action}"
        elif primary.observation_type == ObservationType.REPEATED_ACTION:
            trigger = self._infer_trigger_from_context(merged_context)
            action = primary.corrected_action or primary.description
        elif primary.observation_type == ObservationType.CONSISTENT_CHOICE:
            trigger = f"When choosing between options"
            action = f"Prefer '{primary.corrected_action}'"
        elif primary.observation_type == ObservationType.FAILURE_RECOVERY:
            trigger = f"If '{primary.original_action}' fails"
            action = f"Recover by: {primary.corrected_action}"

        # Auto-generate failure modes
        failure_modes = []
        for obs in observations:
            if obs.observation_type == ObservationType.FAILURE_RECOVERY:
                if obs.original_action:
                    failure_modes.append(obs.original_action)

        return LearnedPattern(
            id=str(uuid.uuid4())[:8],
            trigger=trigger,
            action=action,
            context=merged_context,
            evidence=[o.id for o in observations],
            observation_count=len(observations),
            confidence=0.5 + 0.1 * len(observations),  # Start higher with more evidence
            failure_modes=failure_modes,
        )

    def _infer_trigger_from_context(self, context: Dict[str, Any]) -> str:
        """Infer a trigger description from context."""
        parts = []
        if "app" in context:
            parts.append(f"In {context['app']}")
        if "task" in context:
            parts.append(f"when {context['task']}")
        if "url" in context:
            parts.append(f"on {context['url']}")

        return " ".join(parts) if parts else "In this context"

    # =========================================================================
    # Using Patterns
    # =========================================================================

    def get_patterns_for_context(
        self,
        context: Dict[str, Any],
        min_confidence: float = 0.4,
    ) -> List[LearnedPattern]:
        """
        Get patterns applicable to the given context.

        Args:
            context: Current context (app, task, etc.)
            min_confidence: Minimum confidence threshold

        Returns:
            List of applicable patterns, sorted by confidence
        """
        applicable = []

        for pattern in self.patterns.values():
            if pattern.is_archived:
                continue
            if pattern.confidence < min_confidence:
                continue

            # Check if pattern context matches
            matches = all(
                context.get(k) == v
                for k, v in pattern.context.items()
            )

            if matches or not pattern.context:
                applicable.append(pattern)

        # Sort by confidence
        applicable.sort(key=lambda p: p.confidence, reverse=True)
        return applicable

    def apply_pattern(
        self,
        pattern: LearnedPattern,
        success: bool = True,
    ) -> None:
        """
        Record that a pattern was applied.

        Updates statistics and confidence based on outcome.

        Args:
            pattern: The pattern that was applied
            success: Whether application was successful
        """
        pattern.times_applied += 1
        if success:
            pattern.times_successful += 1
            pattern.confidence = min(1.0, pattern.confidence + 0.02)
        else:
            pattern.confidence = max(0.1, pattern.confidence - 0.05)

        pattern.last_applied = datetime.now()
        self._save_data()

    def get_pattern(self, pattern_id: str) -> Optional[LearnedPattern]:
        """Get a pattern by ID."""
        return self.patterns.get(pattern_id)

    def list_patterns(
        self,
        include_archived: bool = False,
    ) -> List[LearnedPattern]:
        """Get all patterns."""
        if include_archived:
            return list(self.patterns.values())
        return [p for p in self.patterns.values() if not p.is_archived]

    def archive_pattern(self, pattern_id: str) -> bool:
        """Archive a pattern (soft delete)."""
        pattern = self.patterns.get(pattern_id)
        if pattern:
            pattern.is_archived = True
            self._save_data()
            return True
        return False

    def delete_pattern(self, pattern_id: str) -> bool:
        """Permanently delete a pattern."""
        if pattern_id in self.patterns:
            del self.patterns[pattern_id]
            self._save_data()
            return True
        return False

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about learned patterns."""
        active_patterns = [p for p in self.patterns.values() if not p.is_archived]

        by_type = {}
        for obs in self.observations.values():
            t = obs.observation_type.value
            by_type[t] = by_type.get(t, 0) + 1

        return {
            "total_observations": len(self.observations),
            "observations_by_type": by_type,
            "total_patterns": len(active_patterns),
            "archived_patterns": len(self.patterns) - len(active_patterns),
            "avg_confidence": sum(p.confidence for p in active_patterns) / max(len(active_patterns), 1),
            "total_applications": sum(p.times_applied for p in active_patterns),
        }


# Singleton instance
_learner: Optional[PatternLearner] = None


def get_pattern_learner() -> PatternLearner:
    """Get the singleton PatternLearner instance."""
    global _learner
    if _learner is None:
        _learner = PatternLearner()
    return _learner
