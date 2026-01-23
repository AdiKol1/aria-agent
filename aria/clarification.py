"""
Aria Clarification Engine

Decides when to ask user for clarification vs. just trying.
Generates natural clarifying questions when needed.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from .intent import Intent
from .planner import Plan


@dataclass
class Clarification:
    """A clarification to ask the user."""
    question: str  # The question to ask
    options: List[str]  # Possible answers
    reason: str  # Why we're asking
    default: Optional[str] = None  # Default if user doesn't respond


# Destructive actions that should always be confirmed
DESTRUCTIVE_ACTIONS = [
    "delete", "remove", "erase", "clear",
    "send", "post", "publish", "submit",
    "format", "reset", "uninstall",
    "overwrite", "replace"
]

# Words indicating uncertainty
UNCERTAINTY_WORDS = [
    "maybe", "possibly", "might", "could",
    "something", "anything", "whatever",
    "thing", "stuff", "that"
]


class ClarificationEngine:
    """Decides when and what to ask for clarification."""

    def should_ask(
        self,
        intent: Intent,
        plan: Optional[Plan] = None,
        failure_count: int = 0
    ) -> Optional[Clarification]:
        """
        Decide if we should ask for clarification.

        Rules:
        1. Confidence < 0.4 → Ask (too uncertain)
        2. Destructive action → Confirm
        3. Multiple equally-likely options → Ask
        4. Failed 2+ times → Ask for help
        5. Otherwise → Try and adapt

        Args:
            intent: Understood intent
            plan: Generated plan (if available)
            failure_count: How many times we've failed this task

        Returns:
            Clarification to ask, or None if we should just try
        """
        # Rule 4: Too many failures - ask for help
        if failure_count >= 2:
            return Clarification(
                question="I've tried a couple times but couldn't complete this. Can you help me understand what you want?",
                options=["Try again", "Show me what to click", "Do something else"],
                reason="multiple_failures"
            )

        # Rule 1: Very low confidence - ask
        if intent.confidence < 0.4:
            return self._ask_for_clarification(intent)

        # Rule 2: Destructive action - confirm
        if self._is_destructive(intent.goal):
            return Clarification(
                question=f"Just to confirm - you want me to {intent.goal.lower()}?",
                options=["Yes, do it", "No, cancel"],
                reason="destructive_action",
                default="No, cancel"
            )

        # Rule 3: Multiple ambiguous elements - ask
        if len(intent.ambiguous_elements) >= 2:
            return self._ask_about_ambiguity(intent)

        # Otherwise - try and adapt
        return None

    def _is_destructive(self, goal: str) -> bool:
        """Check if the goal involves destructive actions."""
        lower = goal.lower()
        return any(action in lower for action in DESTRUCTIVE_ACTIONS)

    def _ask_for_clarification(self, intent: Intent) -> Clarification:
        """Generate a clarifying question for low-confidence intent."""
        # If we have specific ambiguities, ask about those
        if intent.ambiguous_elements:
            return self._ask_about_ambiguity(intent)

        # If we resolved something but aren't sure, confirm
        if intent.resolved_references:
            refs = list(intent.resolved_references.items())
            ref_text = refs[0]  # First resolution
            return Clarification(
                question=f"When you said '{ref_text[0]}', did you mean {ref_text[1]}?",
                options=["Yes", "No, I meant something else"],
                reason="low_confidence_resolution"
            )

        # General clarification
        return Clarification(
            question=f"I want to make sure I understand. You want me to {intent.goal.lower()}?",
            options=["Yes, that's right", "No, let me explain"],
            reason="low_confidence"
        )

    def _ask_about_ambiguity(self, intent: Intent) -> Clarification:
        """Generate a question about ambiguous elements."""
        if not intent.ambiguous_elements:
            return self._ask_for_clarification(intent)

        ambiguity = intent.ambiguous_elements[0]

        # Generate natural question
        if "which" in ambiguity.lower() or "what" in ambiguity.lower():
            question = ambiguity  # Already a question
        else:
            question = f"I'm not sure about: {ambiguity}. Can you clarify?"

        return Clarification(
            question=question,
            options=["Let me show you", "I'll explain"],
            reason="ambiguity"
        )

    def generate_stuck_question(
        self,
        goal: str,
        what_was_tried: str,
        what_failed: str
    ) -> Clarification:
        """Generate a question when we're stuck."""
        return Clarification(
            question=f"I tried to {what_was_tried}, but {what_failed}. What should I do?",
            options=[
                "Try a different approach",
                "Let me help you",
                "Never mind, skip this"
            ],
            reason="stuck"
        )

    def generate_verification_question(
        self,
        action_taken: str,
        expected_result: str
    ) -> Clarification:
        """Generate a question to verify if action worked."""
        return Clarification(
            question=f"I {action_taken}. Did that work?",
            options=["Yes", "No", "Partially"],
            reason="verification"
        )

    def interpret_response(
        self,
        clarification: Clarification,
        user_response: str
    ) -> Dict[str, Any]:
        """
        Interpret user's response to a clarification.

        Returns dict with:
        - understood: bool - Did we understand the response?
        - action: str - What to do next ("proceed", "retry", "cancel", "explain")
        - details: Any - Additional details from response
        """
        lower = user_response.lower().strip()

        # Check for affirmatives
        if lower in ["yes", "yeah", "yep", "sure", "ok", "okay", "correct", "right"]:
            return {"understood": True, "action": "proceed", "details": None}

        # Check for negatives
        if lower in ["no", "nope", "cancel", "stop", "never mind", "nevermind"]:
            return {"understood": True, "action": "cancel", "details": None}

        # Check for retry
        if "try" in lower or "again" in lower or "retry" in lower:
            return {"understood": True, "action": "retry", "details": None}

        # Check for explanation coming
        if "let me" in lower or "i'll" in lower or "explain" in lower:
            return {"understood": True, "action": "wait_for_explanation", "details": None}

        # Check for showing/pointing
        if "show" in lower or "point" in lower or "here" in lower:
            return {"understood": True, "action": "wait_for_visual", "details": None}

        # Didn't understand - might be the actual answer
        return {
            "understood": False,
            "action": "interpret_as_input",
            "details": user_response
        }


# Singleton
_clarification: Optional[ClarificationEngine] = None


def get_clarification_engine() -> ClarificationEngine:
    """Get singleton ClarificationEngine instance."""
    global _clarification
    if _clarification is None:
        _clarification = ClarificationEngine()
    return _clarification
