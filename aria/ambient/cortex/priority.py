"""
Aria Ambient Intelligence - Priority Calculator

Calculates the priority of insights based on multiple factors.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import logging

from ..models import Signal, Insight, World, Goal
from ..constants import (
    InsightPriority,
    UrgencyLevel,
    GoalStatus,
    GoalPriority,
    PRIORITY_THRESHOLDS,
)

logger = logging.getLogger(__name__)


class PriorityCalculator:
    """
    Calculates priority scores for insights.

    Priority is determined by:
    1. Relevance score (how well signal matches world)
    2. Urgency (time-sensitivity)
    3. Actionability (can user do something?)
    4. Novelty (is this new information?)
    5. Goal alignment (does it relate to active goals?)

    Output is a normalized score from 0.0 to 1.0, plus a
    categorical priority level (critical, high, medium, low).
    """

    def __init__(self, history_window_hours: int = 24):
        """
        Initialize the priority calculator.

        Args:
            history_window_hours: How far back to look for novelty calculation
        """
        self._history_window = history_window_hours
        self._recent_signals: List[Signal] = []  # For novelty calculation

    def calculate(
        self,
        signal: Signal,
        world: World,
        relevance_score: float
    ) -> tuple:
        """
        Calculate priority for a signal in a world context.

        Args:
            signal: The signal to prioritize
            world: The world context
            relevance_score: Pre-calculated relevance score (0-1)

        Returns:
            Tuple of (priority_score: float, priority_level: InsightPriority, urgency: UrgencyLevel)
        """
        # Component scores (all 0-1)
        urgency_score = self.calculate_urgency(signal)
        actionability_score = self.calculate_actionability(signal)
        novelty_score = self.calculate_novelty(signal)
        goal_score = self.calculate_goal_alignment(signal, world)

        # Weighted combination
        weights = {
            "relevance": 0.3,
            "urgency": 0.25,
            "actionability": 0.15,
            "novelty": 0.15,
            "goal_alignment": 0.15,
        }

        priority_score = (
            relevance_score * weights["relevance"] +
            urgency_score * weights["urgency"] +
            actionability_score * weights["actionability"] +
            novelty_score * weights["novelty"] +
            goal_score * weights["goal_alignment"]
        )

        # Determine priority level and urgency
        priority_level = self._score_to_priority(priority_score)
        urgency_level = self._determine_urgency(signal, urgency_score)

        # Track signal for future novelty calculations
        self._record_signal(signal)

        return priority_score, priority_level, urgency_level

    def calculate_urgency(self, signal: Signal) -> float:
        """
        Calculate how time-sensitive a signal is.

        Factors:
        - Signal type (some are inherently more urgent)
        - Expiration time
        - Keywords indicating urgency

        Returns:
            Urgency score (0-1)
        """
        score = 0.5  # Default medium urgency

        # Check signal type urgency
        urgent_types = [
            "deadline_approaching",
            "price_change",
            "calendar_reminder",
            "email_important",
        ]
        if signal.type.value in urgent_types:
            score = max(score, 0.8)

        # Check for deadline/expiration
        if signal.expires_at:
            try:
                expires = datetime.fromisoformat(signal.expires_at)
                hours_until = (expires - datetime.now()).total_seconds() / 3600

                if hours_until < 1:
                    score = max(score, 1.0)
                elif hours_until < 4:
                    score = max(score, 0.9)
                elif hours_until < 24:
                    score = max(score, 0.7)

            except ValueError:
                pass

        # Check content for urgency keywords
        urgency_keywords = [
            "urgent", "immediately", "asap", "deadline",
            "breaking", "alert", "critical", "emergency",
            "now", "today", "expires"
        ]
        text = f"{signal.title} {signal.content}".lower()
        for keyword in urgency_keywords:
            if keyword in text:
                score = max(score, 0.85)
                break

        return min(score, 1.0)

    def calculate_actionability(self, signal: Signal) -> float:
        """
        Calculate how actionable a signal is.

        Signals that allow the user to do something are more valuable
        than purely informational ones.

        Returns:
            Actionability score (0-1)
        """
        score = 0.5

        # Check signal type
        actionable_types = [
            "calendar_event",
            "calendar_reminder",
            "email_important",
            "opportunity_detected",
            "social_mention",
        ]
        if signal.type.value in actionable_types:
            score = max(score, 0.7)

        # Check for action words in content
        action_keywords = [
            "respond", "reply", "review", "approve", "schedule",
            "submit", "buy", "sell", "register", "sign up",
            "click", "call", "email", "follow up"
        ]
        text = f"{signal.title} {signal.content}".lower()
        for keyword in action_keywords:
            if keyword in text:
                score = max(score, 0.8)
                break

        # Check raw_data for action hints
        if signal.raw_data.get("action_required"):
            score = max(score, 0.9)
        if signal.raw_data.get("has_link"):
            score = max(score, 0.6)

        return min(score, 1.0)

    def calculate_novelty(self, signal: Signal) -> float:
        """
        Calculate how novel/new the information is.

        Avoids repeatedly surfacing the same information.

        Returns:
            Novelty score (0-1, higher = more novel)
        """
        # Check against recent signals
        text_key = f"{signal.title}:{signal.url}"

        for recent in self._recent_signals:
            recent_key = f"{recent.title}:{recent.url}"
            if self._is_similar(text_key, recent_key):
                return 0.2  # Low novelty if similar to recent

        # Fresh signal
        return 0.9

    def _is_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two texts are similar (simple comparison)."""
        if text1 == text2:
            return True

        # Jaccard similarity on words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return False

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return (intersection / union) >= threshold if union > 0 else False

    def calculate_goal_alignment(self, signal: Signal, world: World) -> float:
        """
        Calculate how well the signal aligns with active goals.

        Returns:
            Goal alignment score (0-1)
        """
        if not world.goals:
            return 0.5  # Neutral if no goals

        active_goals = [g for g in world.goals if g.status == GoalStatus.ACTIVE]
        if not active_goals:
            return 0.5

        max_alignment = 0.0
        text = f"{signal.title} {signal.content}".lower()

        for goal in active_goals:
            alignment = self._goal_alignment_score(text, goal)
            max_alignment = max(max_alignment, alignment)

        return max_alignment

    def _goal_alignment_score(self, text: str, goal: Goal) -> float:
        """Calculate alignment score for a single goal."""
        score = 0.0

        # Check progress indicators
        for indicator in goal.progress_indicators:
            if indicator.lower() in text:
                score += 0.3

        # Check risk indicators
        for indicator in goal.risk_indicators:
            if indicator.lower() in text:
                score += 0.2  # Risk signals are still relevant

        # Check description keywords
        goal_words = [w for w in goal.description.lower().split() if len(w) > 3]
        for word in goal_words[:10]:  # Limit to first 10 words
            if word in text:
                score += 0.1

        # Priority multiplier
        priority_mult = {
            GoalPriority.CRITICAL: 1.5,
            GoalPriority.HIGH: 1.2,
            GoalPriority.MEDIUM: 1.0,
            GoalPriority.LOW: 0.8,
        }
        score *= priority_mult.get(goal.priority, 1.0)

        return min(score, 1.0)

    def _score_to_priority(self, score: float) -> InsightPriority:
        """Convert numeric score to priority level."""
        if score >= PRIORITY_THRESHOLDS["critical"]:
            return InsightPriority.CRITICAL
        elif score >= PRIORITY_THRESHOLDS["high"]:
            return InsightPriority.HIGH
        elif score >= PRIORITY_THRESHOLDS["medium"]:
            return InsightPriority.MEDIUM
        else:
            return InsightPriority.LOW

    def _determine_urgency(
        self,
        signal: Signal,
        urgency_score: float
    ) -> UrgencyLevel:
        """Determine urgency level from signal and score."""
        if urgency_score >= 0.9:
            return UrgencyLevel.IMMEDIATE
        elif urgency_score >= 0.7:
            return UrgencyLevel.TODAY

        # Check signal type for inherent urgency
        today_types = ["calendar_event", "calendar_reminder", "deadline_approaching"]
        if signal.type.value in today_types:
            return UrgencyLevel.TODAY

        if urgency_score >= 0.5:
            return UrgencyLevel.THIS_WEEK

        return UrgencyLevel.WHENEVER

    def _record_signal(self, signal: Signal) -> None:
        """Record signal for future novelty calculations."""
        self._recent_signals.append(signal)

        # Cleanup old signals
        cutoff = datetime.now() - timedelta(hours=self._history_window)
        self._recent_signals = [
            s for s in self._recent_signals
            if datetime.fromisoformat(s.timestamp) > cutoff
        ]

    def clear_history(self) -> None:
        """Clear signal history."""
        self._recent_signals.clear()
