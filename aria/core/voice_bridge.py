"""Voice Bridge - Connects voice transcription to AriaEngine.

This module provides a bridge between the Gemini voice system and the
new AriaEngine, enabling direct action execution from voice commands
without relying on Gemini's unreliable tool calling.

The key insight is that Gemini's Live API often "confabulates" - it claims
to have executed actions without actually calling the registered tools.
This bridge bypasses that issue by processing transcribed text directly
through the AriaEngine's intent parsing and execution pipeline.

Flow:
    User speaks -> Gemini STT -> VoiceBridge.process_voice_input() ->
    AriaEngine.process() -> Direct action execution

Example:
    from aria.core import VoiceBridge

    bridge = VoiceBridge()

    # When Gemini transcribes text:
    result = bridge.process_voice_input("click on Chrome")
    if result.success:
        print(f"Executed: {result.response}")

    # Check if text should be intercepted before Gemini responds:
    if bridge.should_intercept("open Safari"):
        result = bridge.process_voice_input("open Safari")
"""
import logging
from typing import Optional, Callable, Dict, Any

from .aria_engine import AriaEngine, get_engine
from .ai_handler import AIHandler
from ..intents.base import IntentResult

logger = logging.getLogger(__name__)


class VoiceBridge:
    """Bridges voice transcription to action execution.

    This class receives transcribed text from the Gemini voice system
    and routes it through AriaEngine for reliable action execution.

    The VoiceBridge solves two key problems with Gemini Live API:
    1. Confabulation: Gemini often claims to execute actions without calling tools
    2. Unreliable tool calling: Even when properly configured, tool calls are flaky

    By intercepting transcribed text and processing it through AriaEngine,
    we get reliable, deterministic action execution while still using
    Gemini's excellent speech recognition.

    Attributes:
        _engine: AriaEngine instance for processing commands
        _ai_handler: Optional AIHandler for complex/conversational requests
        _on_action_callback: Called when an action is executed
        _on_response_callback: Called when a response should be spoken

    Example:
        bridge = VoiceBridge()

        # When Gemini transcribes text:
        result = bridge.process_voice_input("click on Chrome")
        if result.success:
            print(f"Executed: {result.response}")
    """

    # Action keywords that indicate the user wants to execute a command
    # NOTE: Avoid ambiguous single words like "tab" (tab key? new tab?) or "enter" (enter key? enter text?)
    # These get false positives from fragmented speech like "No t" → "tab"
    # NOTE: "move" removed - let Gemini handle move_mouse commands since VoiceBridge
    # doesn't have a MOVE intent and can't use coordinate-based movement
    # NOTE: "click", "double click", "right click" removed - let Gemini handle with
    # click_target tool which uses Claude's precise vision for accurate clicking
    ACTION_KEYWORDS = [
        'open', 'type', 'scroll', 'close', 'copy', 'paste',
        'undo', 'save', 'go to', 'navigate', 'launch', 'start',
        'press', 'hit', 'select', 'new tab', 'close tab',
        'switch', 'write', 'input', 'quit', 'exit',
        'drag', 'drop', 'minimize', 'maximize', 'focus'
    ]

    def __init__(self, use_ai_fallback: bool = True):
        """Initialize the VoiceBridge.

        Args:
            use_ai_fallback: Whether to use AIHandler for complex/low-confidence
                           requests. Defaults to True.
        """
        self._engine = get_engine()
        self._ai_handler = AIHandler() if use_ai_fallback else None
        self._on_action_callback: Optional[Callable[[IntentResult], None]] = None
        self._on_response_callback: Optional[Callable[[str], None]] = None

        logger.info(f"VoiceBridge initialized (AI fallback: {use_ai_fallback})")

    def set_callbacks(self,
                     on_action: Optional[Callable[[IntentResult], None]] = None,
                     on_response: Optional[Callable[[str], None]] = None):
        """Set callbacks for action and response events.

        These callbacks allow the voice system to be notified when actions
        are executed or when responses should be spoken.

        Args:
            on_action: Called when an action is executed. Receives IntentResult.
            on_response: Called when a response should be spoken. Receives string.

        Example:
            def handle_action(result):
                print(f"Action executed: {result.response}")

            def handle_response(text):
                # Queue text for text-to-speech
                tts_queue.put(text)

            bridge.set_callbacks(on_action=handle_action, on_response=handle_response)
        """
        self._on_action_callback = on_action
        self._on_response_callback = on_response

    def process_voice_input(self, text: str) -> IntentResult:
        """Process transcribed voice input.

        This is the main entry point for voice command processing.
        It takes transcribed text from speech recognition and routes
        it through AriaEngine for intent parsing and execution.

        Args:
            text: Transcribed text from voice recognition.

        Returns:
            IntentResult with the outcome of the command.
            - success=True: Command was executed successfully
            - success=False: Command failed or was not recognized

        Example:
            result = bridge.process_voice_input("scroll down")
            if result.success:
                print(result.response)  # "Scrolled down"
            else:
                print(f"Failed: {result.error}")
        """
        if not text or not text.strip():
            return IntentResult.error_result(
                error="EMPTY_INPUT",
                response="No voice input received"
            )

        logger.info(f"Processing voice input: {text}")

        try:
            # Use AI fallback if available and configured
            if self._ai_handler:
                result = self._engine.process_with_fallback(
                    text,
                    ai_handler=self._ai_handler.handle
                )
            else:
                result = self._engine.process(text)

            # Notify callbacks
            if result.success and self._on_action_callback:
                self._on_action_callback(result)

            if self._on_response_callback:
                self._on_response_callback(result.response)

            logger.info(f"Voice processing result: success={result.success}, "
                       f"response={result.response[:50]}...")

            return result

        except Exception as e:
            logger.error(f"Error processing voice input: {e}", exc_info=True)
            return IntentResult.error_result(
                error="VOICE_PROCESSING_ERROR",
                response=f"Failed to process voice command: {str(e)}"
            )

    def should_intercept(self, text: str) -> bool:
        """Check if the text looks like an action command.

        This helps decide whether to process the command locally via
        AriaEngine vs letting Gemini handle it conversationally.

        Use this method to determine if voice input should be:
        1. Intercepted and processed immediately (returns True)
        2. Passed through to Gemini for conversational response (returns False)

        Args:
            text: The transcribed text to evaluate.

        Returns:
            True if this looks like an action command that should be
            processed by AriaEngine. False if it appears to be
            conversational and should be handled by Gemini.

        Example:
            # Action commands - intercept these
            bridge.should_intercept("click on Chrome")  # True
            bridge.should_intercept("open Safari")       # True
            bridge.should_intercept("scroll down")       # True

            # Conversational - let Gemini handle
            bridge.should_intercept("what time is it")   # False
            bridge.should_intercept("tell me a joke")    # False
        """
        if not text:
            return False

        # Clean the input first to extract the action command
        from .intent_parser import clean_voice_input
        cleaned = clean_voice_input(text)
        cleaned_lower = cleaned.lower().strip()

        if not cleaned_lower:
            return False

        # Keywords that require a target (incomplete without it)
        keywords_needing_target = {
            'click on': 1,  # needs at least 1 word after "click on"
            'click': 1,     # needs at least 1 word after "click"
            'open': 1,      # needs at least 1 word after "open"
            'type': 1,      # needs at least 1 word after "type"
            'go to': 1,     # needs at least 1 word after "go to"
            'navigate to': 1,
            'search': 1,    # needs at least 1 word after "search"
            'move': 1,      # needs at least 1 word after "move"
        }

        # Check for conversational context markers that indicate this is NOT a command
        # These words suggest the user is talking ABOUT actions, not requesting them
        # Check BOTH the original text and cleaned text (original has more context)
        text_lower = text.lower()
        conversational_markers = [
            'otherwise', "you'll", "you will", "would", "could", "might", "should",
            "don't", "won't", "wouldn't", "couldn't", "shouldn't",
            "if you", "when you", "before you", "after you",
            "did you", "have you", "are you", "were you",
            "why did", "why do", "what if", "instead of",
            "going to", "about to", "trying to",
        ]
        for marker in conversational_markers:
            # Check in original text (more context)
            if marker in text_lower:
                logger.debug(f"Conversational context detected ('{marker}'): {text[:50]}")
                return False

        # Check if the cleaned text STARTS with an action keyword
        # This prevents false matches like "right below the start"
        for keyword in self.ACTION_KEYWORDS:
            # Check if text starts with the keyword (with word boundary)
            if cleaned_lower.startswith(keyword + ' ') or cleaned_lower == keyword:
                # Check if this keyword needs a target
                rest_after_keyword = cleaned_lower[len(keyword):].strip()

                # Check for "keyword on" pattern (like "click on")
                keyword_on = keyword + ' on'
                if cleaned_lower.startswith(keyword_on):
                    rest_after_on = cleaned_lower[len(keyword_on):].strip()
                    # Needs at least one word after "on"
                    if not rest_after_on or rest_after_on in ['the', 'a', 'an', 'my', 'your', 'this', 'that']:
                        logger.debug(f"Incomplete command '{keyword} on' - waiting for target")
                        return False

                # Check if keyword needs a target
                if keyword in keywords_needing_target:
                    min_words = keywords_needing_target[keyword]
                    words_after = rest_after_keyword.split()
                    # Filter out articles/prepositions that don't count as targets
                    meaningful_words = [w for w in words_after if w not in ['the', 'a', 'an', 'on', 'to', 'in', 'at', 'my', 'your']]
                    if len(meaningful_words) < min_words:
                        logger.debug(f"Incomplete command '{keyword}' - waiting for target (has {len(meaningful_words)} words, need {min_words})")
                        return False

                logger.debug(f"Text starts with action keyword '{keyword}': {cleaned}")
                return True
            # Also check for patterns like "keyword something" at start
            if cleaned_lower.startswith(keyword):
                # Make sure it's a word boundary, not part of another word
                rest = cleaned_lower[len(keyword):]
                if not rest or rest[0] in ' \t\n.,!?':
                    logger.debug(f"Text starts with action keyword '{keyword}': {cleaned}")
                    return True

        # Parse and check confidence for edge cases
        try:
            from .intent_parser import parse

            intent = parse(text)

            # High confidence non-conversation intent = intercept
            if intent.confidence > 0.6 and not intent.requires_ai:
                logger.debug(f"High confidence intent ({intent.confidence}): {text}")
                return True

        except Exception as e:
            logger.warning(f"Error parsing for interception check: {e}")

        return False

    def process_if_action(self, text: str) -> Optional[IntentResult]:
        """Process text only if it appears to be an action command.

        Convenience method that combines should_intercept() and
        process_voice_input(). Returns None if the text doesn't
        look like an action command.

        Args:
            text: The transcribed text to evaluate and potentially process.

        Returns:
            IntentResult if this was an action command, None otherwise.

        Example:
            result = bridge.process_if_action(text)
            if result is not None:
                # We handled it locally
                print(f"Executed: {result.response}")
            else:
                # Let Gemini handle it conversationally
                pass
        """
        if self.should_intercept(text):
            return self.process_voice_input(text)
        return None

    def get_engine(self) -> AriaEngine:
        """Get the underlying AriaEngine instance.

        Useful for advanced use cases where direct engine access is needed.

        Returns:
            The AriaEngine instance used by this bridge.
        """
        return self._engine

    def get_capabilities(self) -> Dict[str, Any]:
        """Get information about the bridge's capabilities.

        Returns a dictionary describing what commands the bridge can handle,
        useful for debugging or capability discovery.

        Returns:
            Dictionary with:
            - action_keywords: List of keywords that trigger interception
            - engine_capabilities: Full capability info from AriaEngine
            - ai_fallback_enabled: Whether AI fallback is configured
        """
        return {
            "action_keywords": self.ACTION_KEYWORDS.copy(),
            "engine_capabilities": self._engine.get_capabilities(),
            "ai_fallback_enabled": self._ai_handler is not None
        }


# ============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ============================================================================

# Singleton bridge instance
_bridge: Optional[VoiceBridge] = None


def get_voice_bridge(use_ai_fallback: bool = True) -> VoiceBridge:
    """Get the shared VoiceBridge instance.

    Args:
        use_ai_fallback: Whether to use AI fallback (only applies on first call).

    Returns:
        Singleton VoiceBridge instance.

    Example:
        from aria.core.voice_bridge import get_voice_bridge

        bridge = get_voice_bridge()
        result = bridge.process_voice_input("open Chrome")
    """
    global _bridge
    if _bridge is None:
        _bridge = VoiceBridge(use_ai_fallback=use_ai_fallback)
    return _bridge


def process_voice(text: str) -> IntentResult:
    """Process voice input using the default bridge.

    Convenience function for simple one-off processing.

    Args:
        text: Transcribed voice input.

    Returns:
        IntentResult with execution outcome.

    Example:
        from aria.core.voice_bridge import process_voice

        result = process_voice("scroll down")
        print(result.response)
    """
    return get_voice_bridge().process_voice_input(text)


def should_intercept_voice(text: str) -> bool:
    """Check if voice input should be intercepted.

    Convenience function using the default bridge.

    Args:
        text: Transcribed voice input.

    Returns:
        True if this looks like an action command.

    Example:
        from aria.core.voice_bridge import should_intercept_voice

        if should_intercept_voice(text):
            # Process locally
            result = process_voice(text)
        else:
            # Let Gemini handle conversationally
            pass
    """
    return get_voice_bridge().should_intercept(text)
