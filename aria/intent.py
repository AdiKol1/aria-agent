"""
Aria Intent Understanding Engine

Resolves vague user inputs into concrete goals using memory context.
"Go to that thing" + memory → "Open ClubEd website"
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from .config import ANTHROPIC_API_KEY, CLAUDE_MODEL_FAST
from .lazy_anthropic import get_client as get_anthropic_client


@dataclass
class Intent:
    """Represents understood user intent."""
    goal: str  # Concrete goal: "Open ClubEd in Chrome"
    original_input: str  # What user actually said
    confidence: float  # 0-1 confidence score
    resolved_references: Dict[str, str] = field(default_factory=dict)  # "that thing" → "ClubEd"
    ambiguous_elements: List[str] = field(default_factory=list)  # Things we're unsure about
    requires_screen: bool = True  # Does this need visual context?
    is_simple: bool = False  # Can be done in 1 action?
    suggested_action: Optional[str] = None  # Immediate action if simple


INTENT_SYSTEM_PROMPT = """You are an intent understanding system. Your job is to figure out what the user ACTUALLY wants, using their memory/context to resolve vague references.

## YOUR TASK
Given user input and their memory context, determine:
1. GOAL: What concrete action do they want? Be specific.
2. CONFIDENCE: How sure are you? (0.0 to 1.0)
3. RESOLVED: What vague terms did you resolve using memory?
4. AMBIGUOUS: What's still unclear?
5. SIMPLE: Can this be done in ONE action (click, open app, etc)?
6. NEEDS_SCREEN: Do we need to see the screen to do this?

## RESOLUTION RULES
- "that thing/app/page" → Look in memory for recent/relevant items
- "our page/site" → User's business page (from memory)
- "the browser" → User's preferred browser (from memory)
- Pronouns (it, this, that) → Most recently discussed item

## EXAMPLES

Input: "go to that learning thing"
Memory: "User has ClubEd (lifelong learning platform) in shortcuts"
Output: {
  "goal": "Open ClubEd website",
  "confidence": 0.85,
  "resolved": {"that learning thing": "ClubEd"},
  "ambiguous": [],
  "simple": true,
  "needs_screen": false,
  "suggested_action": {"action": "open_url", "url": "https://clubed.com"}
}

Input: "click the blue button"
Memory: (none relevant)
Output: {
  "goal": "Click a blue button on screen",
  "confidence": 0.6,
  "resolved": {},
  "ambiguous": ["which blue button - need to see screen"],
  "simple": true,
  "needs_screen": true,
  "suggested_action": null
}

Input: "open our facebook page"
Memory: "User manages ClubEd Facebook page", "User only uses Chrome"
Output: {
  "goal": "Open ClubEd Facebook page in Chrome",
  "confidence": 0.9,
  "resolved": {"our facebook page": "ClubEd Facebook page"},
  "ambiguous": [],
  "simple": false,
  "needs_screen": true,
  "suggested_action": null
}

Respond with ONLY valid JSON, no other text."""


class IntentEngine:
    """Understands user intent using memory context."""

    def __init__(self):
        self.client = get_anthropic_client(ANTHROPIC_API_KEY)
        self._cache: Dict[str, Intent] = {}  # Simple cache for repeated queries
        self._cache_ttl = 60  # Cache for 60 seconds

    def understand(
        self,
        user_input: str,
        memory_facts: List[str],
        recent_context: Optional[str] = None
    ) -> Intent:
        """
        Understand user intent using memory to resolve references.

        Args:
            user_input: What the user said
            memory_facts: Relevant facts from memory
            recent_context: Recent conversation context

        Returns:
            Intent object with resolved goal and confidence
        """
        # Check cache first
        cache_key = f"{user_input}:{hash(tuple(memory_facts))}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build context for Claude
        memory_context = "\n".join(f"- {fact}" for fact in memory_facts) if memory_facts else "No relevant memories"

        prompt = f"""User said: "{user_input}"

Memory context:
{memory_context}

{f"Recent conversation: {recent_context}" if recent_context else ""}

What is the user's intent? Respond with JSON only."""

        try:
            response = self.client.messages.create(
                model=CLAUDE_MODEL_FAST,  # Use Haiku for speed
                max_tokens=500,
                system=INTENT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response
            result = self._parse_response(response.content[0].text, user_input)

            # Cache result
            self._cache[cache_key] = result

            return result

        except Exception as e:
            print(f"Intent understanding error: {e}")
            # Return low-confidence fallback
            return Intent(
                goal=user_input,
                original_input=user_input,
                confidence=0.3,
                requires_screen=True,
                is_simple=False
            )

    def _parse_response(self, response_text: str, original_input: str) -> Intent:
        """Parse Claude's JSON response into Intent object."""
        try:
            # Try to extract JSON
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            data = json.loads(text)

            return Intent(
                goal=data.get("goal", original_input),
                original_input=original_input,
                confidence=float(data.get("confidence", 0.5)),
                resolved_references=data.get("resolved", {}),
                ambiguous_elements=data.get("ambiguous", []),
                requires_screen=data.get("needs_screen", True),
                is_simple=data.get("simple", False),
                suggested_action=data.get("suggested_action")
            )

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse intent response: {e}")
            return Intent(
                goal=original_input,
                original_input=original_input,
                confidence=0.4,
                requires_screen=True,
                is_simple=False
            )

    def quick_classify(self, user_input: str) -> Dict[str, Any]:
        """
        Quick classification without API call.
        For obvious cases like greetings, exits, simple commands.
        """
        lower = user_input.lower().strip()

        # Exit phrases
        if lower in ["bye", "goodbye", "exit", "quit", "done", "stop"]:
            return {"type": "exit", "confidence": 1.0}

        # Greetings
        if lower in ["hi", "hello", "hey", "hey aria", "hi aria"]:
            return {"type": "greeting", "confidence": 1.0}

        # Affirmations
        if lower in ["yes", "yeah", "yep", "sure", "ok", "okay", "correct"]:
            return {"type": "affirmation", "confidence": 1.0}

        # Negations
        if lower in ["no", "nope", "cancel", "never mind", "nevermind"]:
            return {"type": "negation", "confidence": 1.0}

        # Simple commands (high confidence patterns)
        simple_patterns = {
            "scroll down": {"action": "scroll", "amount": -300, "done": True},
            "scroll up": {"action": "scroll", "amount": 300, "done": True},
            "press enter": {"action": "press", "key": "enter", "done": True},
            "press escape": {"action": "press", "key": "escape", "done": True},
            "press tab": {"action": "press", "key": "tab", "done": True},
        }

        for pattern, action in simple_patterns.items():
            if pattern in lower:
                return {
                    "type": "simple_action",
                    "action": action,
                    "confidence": 0.95
                }

        # Needs full understanding
        return {"type": "complex", "confidence": 0.0}

    def clear_cache(self):
        """Clear the intent cache."""
        self._cache.clear()


# Singleton
_intent_engine: Optional[IntentEngine] = None


def get_intent_engine() -> IntentEngine:
    """Get singleton IntentEngine instance."""
    global _intent_engine
    if _intent_engine is None:
        _intent_engine = IntentEngine()
    return _intent_engine
