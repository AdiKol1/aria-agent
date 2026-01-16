"""
Aria Learning System

Tracks task outcomes and learns from success/failure patterns.
Enables Aria to improve over time.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from .config import DATA_PATH
from .planner import Plan, PlanStep


# Learning data storage
LEARNING_PATH = DATA_PATH / "learning"
LEARNING_PATH.mkdir(parents=True, exist_ok=True)


@dataclass
class TaskOutcome:
    """Record of a task execution."""
    task_id: str
    goal: str
    success: bool
    duration_ms: int
    steps_attempted: int
    steps_succeeded: int
    failure_reason: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    approach_used: Optional[str] = None  # Which procedure/plan


@dataclass
class ApproachStats:
    """Statistics for a particular approach to a task type."""
    approach_id: str
    task_pattern: str  # e.g., "open_website", "navigate_to_page"
    success_count: int = 0
    failure_count: int = 0
    total_duration_ms: int = 0
    last_used: Optional[str] = None
    last_failure_reason: Optional[str] = None

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    @property
    def avg_duration_ms(self) -> int:
        total = self.success_count + self.failure_count
        return self.total_duration_ms // total if total > 0 else 2000


class LearningEngine:
    """Tracks outcomes and learns from experience."""

    def __init__(self):
        self.outcomes_file = LEARNING_PATH / "outcomes.json"
        self.approaches_file = LEARNING_PATH / "approaches.json"
        self.patterns_file = LEARNING_PATH / "patterns.json"

        self.outcomes: List[TaskOutcome] = []
        self.approaches: Dict[str, ApproachStats] = {}
        self.patterns: Dict[str, Dict] = {}  # Learned patterns

        self._load_data()

    def _load_data(self):
        """Load learning data from files."""
        # Load outcomes
        if self.outcomes_file.exists():
            try:
                with open(self.outcomes_file) as f:
                    data = json.load(f)
                    self.outcomes = [TaskOutcome(**o) for o in data[-100:]]  # Keep last 100
            except Exception as e:
                print(f"Error loading outcomes: {e}")

        # Load approaches
        if self.approaches_file.exists():
            try:
                with open(self.approaches_file) as f:
                    data = json.load(f)
                    self.approaches = {
                        k: ApproachStats(**v) for k, v in data.items()
                    }
            except Exception as e:
                print(f"Error loading approaches: {e}")

        # Load patterns
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file) as f:
                    self.patterns = json.load(f)
            except Exception as e:
                print(f"Error loading patterns: {e}")

    def _save_data(self):
        """Save learning data to files."""
        try:
            # Save outcomes (keep last 100)
            with open(self.outcomes_file, 'w') as f:
                json.dump([
                    {
                        'task_id': o.task_id,
                        'goal': o.goal,
                        'success': o.success,
                        'duration_ms': o.duration_ms,
                        'steps_attempted': o.steps_attempted,
                        'steps_succeeded': o.steps_succeeded,
                        'failure_reason': o.failure_reason,
                        'timestamp': o.timestamp,
                        'approach_used': o.approach_used
                    }
                    for o in self.outcomes[-100:]
                ], f, indent=2)

            # Save approaches
            with open(self.approaches_file, 'w') as f:
                json.dump({
                    k: {
                        'approach_id': v.approach_id,
                        'task_pattern': v.task_pattern,
                        'success_count': v.success_count,
                        'failure_count': v.failure_count,
                        'total_duration_ms': v.total_duration_ms,
                        'last_used': v.last_used,
                        'last_failure_reason': v.last_failure_reason
                    }
                    for k, v in self.approaches.items()
                }, f, indent=2)

            # Save patterns
            with open(self.patterns_file, 'w') as f:
                json.dump(self.patterns, f, indent=2)

        except Exception as e:
            print(f"Error saving learning data: {e}")

    def record_outcome(
        self,
        goal: str,
        success: bool,
        duration_ms: int,
        steps_attempted: int,
        steps_succeeded: int,
        failure_reason: Optional[str] = None,
        approach_id: Optional[str] = None,
        plan: Optional[Plan] = None
    ):
        """
        Record the outcome of a task execution.

        Args:
            goal: What we tried to do
            success: Did it work?
            duration_ms: How long it took
            steps_attempted: How many steps we tried
            steps_succeeded: How many worked
            failure_reason: Why it failed (if applicable)
            approach_id: Which approach/procedure we used
            plan: The plan we executed
        """
        import hashlib
        task_id = hashlib.md5(f"{goal}{datetime.now().isoformat()}".encode()).hexdigest()[:8]

        outcome = TaskOutcome(
            task_id=task_id,
            goal=goal,
            success=success,
            duration_ms=duration_ms,
            steps_attempted=steps_attempted,
            steps_succeeded=steps_succeeded,
            failure_reason=failure_reason,
            approach_used=approach_id
        )

        self.outcomes.append(outcome)

        # Update approach statistics
        if approach_id:
            self._update_approach(approach_id, goal, success, duration_ms, failure_reason)

        # Learn patterns from success
        if success and plan and len(plan.steps) > 1:
            self._learn_pattern(goal, plan)

        # Save periodically (every 5 outcomes)
        if len(self.outcomes) % 5 == 0:
            self._save_data()

        print(f"Recorded outcome: {goal} - {'SUCCESS' if success else 'FAILED'} ({duration_ms}ms)")

    def _update_approach(
        self,
        approach_id: str,
        goal: str,
        success: bool,
        duration_ms: int,
        failure_reason: Optional[str]
    ):
        """Update statistics for an approach."""
        # Determine task pattern from goal
        task_pattern = self._goal_to_pattern(goal)

        if approach_id not in self.approaches:
            self.approaches[approach_id] = ApproachStats(
                approach_id=approach_id,
                task_pattern=task_pattern
            )

        stats = self.approaches[approach_id]
        stats.last_used = datetime.now().isoformat()
        stats.total_duration_ms += duration_ms

        if success:
            stats.success_count += 1
        else:
            stats.failure_count += 1
            stats.last_failure_reason = failure_reason

    def _goal_to_pattern(self, goal: str) -> str:
        """Convert a specific goal to a general pattern."""
        lower = goal.lower()

        if "open" in lower and ("url" in lower or "website" in lower or "http" in lower):
            return "open_website"
        elif "open" in lower and "app" in lower:
            return "open_app"
        elif "click" in lower:
            return "click_element"
        elif "scroll" in lower:
            return "scroll"
        elif "type" in lower or "enter" in lower:
            return "type_text"
        elif "facebook" in lower:
            return "facebook_navigation"
        elif "navigate" in lower or "go to" in lower:
            return "navigation"
        else:
            return "general"

    def _learn_pattern(self, goal: str, plan: Plan):
        """Learn a successful pattern for future use."""
        pattern = self._goal_to_pattern(goal)

        if pattern not in self.patterns:
            self.patterns[pattern] = {
                "examples": [],
                "common_steps": []
            }

        # Store this successful execution as an example
        example = {
            "goal": goal,
            "steps": [
                {
                    "action": step.action,
                    "description": step.description
                }
                for step in plan.steps
            ],
            "timestamp": datetime.now().isoformat()
        }

        self.patterns[pattern]["examples"].append(example)

        # Keep only last 10 examples per pattern
        self.patterns[pattern]["examples"] = self.patterns[pattern]["examples"][-10:]

    def get_best_approach(self, goal: str) -> Optional[Dict]:
        """
        Get the best known approach for a goal.

        Returns:
            Dict with approach details, or None if no good approach known
        """
        pattern = self._goal_to_pattern(goal)

        # Find approaches for this pattern with good success rate
        best_approach = None
        best_rate = 0.5  # Minimum threshold

        for approach_id, stats in self.approaches.items():
            if stats.task_pattern == pattern and stats.success_rate > best_rate:
                best_approach = stats
                best_rate = stats.success_rate

        if best_approach:
            # Also get examples from patterns
            examples = self.patterns.get(pattern, {}).get("examples", [])
            return {
                "approach_id": best_approach.approach_id,
                "success_rate": best_approach.success_rate,
                "avg_duration_ms": best_approach.avg_duration_ms,
                "examples": examples[-3:]  # Last 3 successful examples
            }

        return None

    def should_try_different_approach(self, approach_id: str) -> bool:
        """
        Check if we should try a different approach based on failure history.

        Returns True if current approach has been failing.
        """
        if approach_id not in self.approaches:
            return False

        stats = self.approaches[approach_id]

        # If success rate is below 30%, try something different
        if stats.success_rate < 0.3 and stats.failure_count >= 3:
            return True

        # If last 3 attempts failed, try something different
        recent_outcomes = [
            o for o in self.outcomes[-10:]
            if o.approach_used == approach_id
        ]
        if len(recent_outcomes) >= 3:
            recent_failures = sum(1 for o in recent_outcomes[-3:] if not o.success)
            if recent_failures >= 3:
                return True

        return False

    def get_failure_insights(self, goal: str) -> List[str]:
        """
        Get insights from past failures for similar goals.

        Returns list of things to avoid or watch out for.
        """
        pattern = self._goal_to_pattern(goal)
        insights = []

        # Look at recent failures for this pattern
        for outcome in reversed(self.outcomes):
            if not outcome.success:
                outcome_pattern = self._goal_to_pattern(outcome.goal)
                if outcome_pattern == pattern and outcome.failure_reason:
                    insights.append(outcome.failure_reason)
                    if len(insights) >= 3:
                        break

        return list(set(insights))  # Deduplicate

    def get_success_rate(self, pattern: Optional[str] = None) -> float:
        """Get overall or pattern-specific success rate."""
        if pattern:
            relevant = [o for o in self.outcomes if self._goal_to_pattern(o.goal) == pattern]
        else:
            relevant = self.outcomes

        if not relevant:
            return 0.5  # No data, assume 50%

        return sum(1 for o in relevant if o.success) / len(relevant)

    def suggest_improvements(self) -> List[Dict]:
        """
        Analyze patterns and suggest improvements.

        Returns list of suggestions based on failure analysis.
        """
        suggestions = []

        # Find patterns with low success rates
        pattern_stats = {}
        for outcome in self.outcomes:
            pattern = self._goal_to_pattern(outcome.goal)
            if pattern not in pattern_stats:
                pattern_stats[pattern] = {"success": 0, "total": 0}
            pattern_stats[pattern]["total"] += 1
            if outcome.success:
                pattern_stats[pattern]["success"] += 1

        for pattern, stats in pattern_stats.items():
            if stats["total"] >= 5:  # Enough data
                rate = stats["success"] / stats["total"]
                if rate < 0.5:
                    suggestions.append({
                        "pattern": pattern,
                        "success_rate": rate,
                        "suggestion": f"Consider improving '{pattern}' tasks - only {rate:.0%} success rate"
                    })

        return suggestions


# Singleton
_learning: Optional[LearningEngine] = None


def get_learning_engine() -> LearningEngine:
    """Get singleton LearningEngine instance."""
    global _learning
    if _learning is None:
        _learning = LearningEngine()
    return _learning
