"""
Multi-Model Reasoning Engine for Aria

Combines the intelligence of Claude, Gemini, and ChatGPT to provide
superior reasoning by synthesizing multiple perspectives.

NOTE: Uses lazy imports via lazy_anthropic module.
"""

import os
import asyncio
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

from .lazy_anthropic import get_client as get_anthropic_client

# Lazy import flags for optional models
_genai = None
_openai_client = None
GEMINI_AVAILABLE = False
OPENAI_AVAILABLE = False


class RequestType(Enum):
    """Type of user request."""
    QUESTION = "question"           # Asking for information
    CONVERSATION = "conversation"   # General chat
    OPINION = "opinion"             # Seeking advice/opinion
    EXPLANATION = "explanation"     # Wants something explained
    ACTION = "action"               # Wants Aria to DO something
    CONFIRMATION = "confirmation"   # Confirming something
    CODING = "coding"               # Code-related request
    UNKNOWN = "unknown"


@dataclass
class ClassifiedRequest:
    """Result of classifying a user request."""
    type: RequestType
    confidence: float
    requires_action: bool
    requires_screen: bool
    summary: str  # What the user actually wants


@dataclass
class ReasoningResult:
    """Result from multi-model reasoning."""
    response: str
    confidence: float
    models_used: List[str]
    reasoning_path: str  # How we arrived at this answer


class MultiModelReasoner:
    """
    Combines multiple AI models for superior reasoning.

    Uses Claude as the primary model, with optional Gemini and GPT
    for additional perspectives on complex questions.

    NOTE: Uses lazy initialization to avoid import blocking.
    """

    def __init__(self, claude_client=None):
        """
        Initialize reasoner.

        Args:
            claude_client: Optional pre-initialized Anthropic client
                          (avoids double-importing anthropic module)
        """
        # Use provided client or lazy-load later
        self._claude = claude_client
        self._gemini = None
        self._openai = None

        self.gemini_available = False
        self.gpt_available = False

        if claude_client:
            print(f"MultiModelReasoner created (using provided Claude client)")
        else:
            print(f"MultiModelReasoner created (lazy init)")

    def _ensure_claude(self):
        """Lazy initialize Claude client."""
        if self._claude is None:
            self._claude = get_anthropic_client()
            print("  - Claude: initialized")
        return self._claude

    def _ensure_gemini(self):
        """Lazy initialize Gemini client."""
        global _genai, GEMINI_AVAILABLE
        if self._gemini is None and not self.gemini_available:
            if os.getenv("GOOGLE_API_KEY"):
                try:
                    import google.generativeai as genai
                    _genai = genai
                    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                    self._gemini = genai.GenerativeModel("gemini-1.5-flash")
                    self.gemini_available = True
                    GEMINI_AVAILABLE = True
                    print("  - Gemini: initialized")
                except Exception as e:
                    print(f"  - Gemini: failed ({e})")
        return self._gemini

    def _ensure_gpt(self):
        """Lazy initialize GPT client."""
        global _openai_client, OPENAI_AVAILABLE
        if self._openai is None and not self.gpt_available:
            if os.getenv("OPENAI_API_KEY"):
                try:
                    import openai
                    self._openai = openai.OpenAI(
                        api_key=os.getenv("OPENAI_API_KEY")
                    )
                    self.gpt_available = True
                    OPENAI_AVAILABLE = True
                    print("  - GPT: initialized")
                except Exception as e:
                    print(f"  - GPT: failed ({e})")
        return self._openai

    def classify_request(self, user_input: str, memory_context: str = "") -> ClassifiedRequest:
        """
        Intelligently classify what type of request this is.

        This determines whether we respond conversationally or take action.
        """
        # Use Claude for classification (fast model)
        classification_prompt = f"""Analyze this user request and classify it.

User said: "{user_input}"
{f"Memory context: {memory_context[:500]}" if memory_context else ""}

Classify as ONE of:
- QUESTION: User is asking for information (who, what, where, when, why, how, is, are, can, do, does, etc.)
- CONVERSATION: General chat, greetings, small talk
- OPINION: User wants advice or your opinion
- EXPLANATION: User wants something explained in detail
- ACTION: User wants you to DO something on their computer (click, type, open, scroll, etc.)
- CONFIRMATION: User is confirming/acknowledging something
- CODING: Code/programming related request

Respond in this exact format:
TYPE: <type>
CONFIDENCE: <0.0-1.0>
REQUIRES_ACTION: <true/false>
REQUIRES_SCREEN: <true/false>
SUMMARY: <what user actually wants in 1 sentence>

Important rules:
- If user asks "what can you do" or "what are your capabilities" - this is QUESTION
- If user asks a question starting with who/what/where/when/why/how - this is QUESTION
- If user says something incomplete like "your process..." - this is QUESTION (they want clarification)
- ONLY classify as ACTION if user explicitly wants you to DO something (click, open, scroll, type)
- Conversational responses like "I can help with X" are NOT actions"""

        try:
            claude = self._ensure_claude()
            response = claude.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=200,
                messages=[{"role": "user", "content": classification_prompt}]
            )
            text = response.content[0].text.strip()

            # Parse response
            lines = text.split("\n")
            result = {}
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    result[key.strip().upper()] = value.strip()

            req_type = RequestType.QUESTION  # Default
            for rt in RequestType:
                if rt.value.upper() == result.get("TYPE", "").upper():
                    req_type = rt
                    break

            return ClassifiedRequest(
                type=req_type,
                confidence=float(result.get("CONFIDENCE", 0.7)),
                requires_action=result.get("REQUIRES_ACTION", "false").lower() == "true",
                requires_screen=result.get("REQUIRES_SCREEN", "false").lower() == "true",
                summary=result.get("SUMMARY", user_input)
            )

        except Exception as e:
            print(f"Classification error: {e}")
            # Default to question (safer - won't try to execute actions)
            return ClassifiedRequest(
                type=RequestType.QUESTION,
                confidence=0.5,
                requires_action=False,
                requires_screen=False,
                summary=user_input
            )

    async def _get_claude_response(self, prompt: str, context: str = "") -> str:
        """Get response from Claude."""
        try:
            messages = [{"role": "user", "content": prompt}]
            if context:
                messages.insert(0, {"role": "user", "content": f"Context: {context}"})
                messages.insert(1, {"role": "assistant", "content": "I understand the context. I'll consider this in my response."})

            claude = self._ensure_claude()
            response = claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system="""You are Aria, an intelligent AI assistant. You are helpful, knowledgeable, and conversational.

Your personality:
- Warm and friendly but not overly casual
- Confident and knowledgeable
- Direct and concise but thorough when needed
- You enjoy helping and explaining things

Important: Respond naturally like an intelligent AI assistant. Have real conversations.
Don't just describe what you can do - actually engage with what the user said.""",
                messages=messages
            )
            return response.content[0].text
        except Exception as e:
            return f"Error getting Claude response: {e}"

    async def _get_gemini_response(self, prompt: str) -> Optional[str]:
        """Get response from Gemini if available."""
        gemini = self._ensure_gemini()
        if gemini is None:
            return None
        try:
            response = await asyncio.to_thread(
                gemini.generate_content, prompt
            )
            return response.text
        except Exception as e:
            print(f"Gemini error: {e}")
            return None

    async def _get_gpt_response(self, prompt: str) -> Optional[str]:
        """Get response from GPT if available."""
        openai_client = self._ensure_gpt()
        if openai_client is None:
            return None
        try:
            response = await asyncio.to_thread(
                lambda: openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500
                )
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"GPT error: {e}")
            return None

    async def reason(
        self,
        user_input: str,
        memory_context: str = "",
        use_multi_model: bool = True
    ) -> ReasoningResult:
        """
        Generate an intelligent response using multi-model reasoning.

        For simple questions, uses Claude alone.
        For complex questions, synthesizes perspectives from multiple models.
        """
        context = f"""You know the following about the user:
{memory_context if memory_context else "No prior context."}

Respond naturally and helpfully to: {user_input}"""

        models_used = ["claude"]

        # Always get Claude's response
        claude_response = await self._get_claude_response(user_input, memory_context)

        # For complex questions, get additional perspectives
        if use_multi_model and (self.gemini_available or self.gpt_available):
            # Check if question is complex enough to warrant multi-model
            complexity_indicators = [
                "explain", "compare", "analyze", "why", "how does",
                "what do you think", "opinion", "best way", "should I"
            ]
            is_complex = any(ind in user_input.lower() for ind in complexity_indicators)

            if is_complex:
                # Get other perspectives in parallel
                tasks = []
                if self.gemini_available:
                    tasks.append(self._get_gemini_response(user_input))
                if self.gpt_available:
                    tasks.append(self._get_gpt_response(user_input))

                if tasks:
                    other_responses = await asyncio.gather(*tasks)

                    # Synthesize responses if we got multiple
                    valid_responses = [r for r in other_responses if r]
                    if valid_responses:
                        if self.gemini_available and other_responses[0]:
                            models_used.append("gemini")
                        if self.gpt_available and other_responses[-1]:
                            models_used.append("gpt")

                        # Have Claude synthesize all perspectives
                        synthesis_prompt = f"""The user asked: "{user_input}"

I have perspectives from multiple AI models:

Claude's view: {claude_response}

{f"Gemini's view: {other_responses[0]}" if self.gemini_available and other_responses[0] else ""}
{f"GPT's view: {other_responses[-1]}" if self.gpt_available and other_responses[-1] else ""}

Synthesize these into a single, comprehensive response that takes the best from each.
Speak as yourself (Aria) - don't mention that you consulted other models.
Be natural and conversational."""

                        claude_response = await self._get_claude_response(synthesis_prompt)

        return ReasoningResult(
            response=claude_response,
            confidence=0.9 if len(models_used) > 1 else 0.8,
            models_used=models_used,
            reasoning_path="multi-model synthesis" if len(models_used) > 1 else "direct response"
        )

    def reason_sync(
        self,
        user_input: str,
        memory_context: str = "",
        use_multi_model: bool = True
    ) -> ReasoningResult:
        """Synchronous wrapper for reason()."""
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(
                self.reason(user_input, memory_context, use_multi_model)
            )
        except RuntimeError:
            return asyncio.run(
                self.reason(user_input, memory_context, use_multi_model)
            )


# Singleton
_reasoner: Optional[MultiModelReasoner] = None


def get_reasoner(claude_client=None) -> MultiModelReasoner:
    """
    Get the singleton MultiModelReasoner instance.

    Args:
        claude_client: Optional pre-initialized Anthropic client to reuse
    """
    global _reasoner
    if _reasoner is None:
        _reasoner = MultiModelReasoner(claude_client=claude_client)
    elif claude_client and _reasoner._claude is None:
        # Update with provided client if we don't have one yet
        _reasoner._claude = claude_client
    return _reasoner
