"""Aria Engine - Main processing pipeline.

This module provides the high-level API for processing user requests.
It orchestrates the complete flow from raw text input to executed action:

1. Normalizes speech input (handling fragmented/misrecognized text)
2. Parses natural language into structured Intent objects
3. Routes intents to the CommandExecutor for execution
4. Returns IntentResult with outcome and response

The AriaEngine is the primary interface for integrating Aria's natural
language processing capabilities into applications.

Example:
    from aria.core import AriaEngine

    engine = AriaEngine()

    # Process a simple command
    result = engine.process("click on Chrome")
    print(result.response)  # "Clicked on Google Chrome"

    # Process with AI fallback for complex cases
    result = engine.process_with_fallback("find all files modified today")
    if not result.success:
        print(result.error)  # May indicate AI assistance needed
"""
import logging
from typing import Optional, Dict, Any, List

from .intent_parser import parse, normalize_speech
from .executor import CommandExecutor
from ..intents.base import Intent, IntentType, IntentResult

# Set up logging
logger = logging.getLogger(__name__)


class AriaEngine:
    """Main Aria processing engine.

    This class provides the primary interface for processing user requests.
    It normalizes input, parses intents, and executes actions through the
    accessibility layer.

    The engine supports two processing modes:
    1. Standard processing - For commands that can be directly executed
    2. Processing with fallback - For commands that may require AI assistance

    Attributes:
        _executor: CommandExecutor instance for executing intents

    Example:
        engine = AriaEngine()

        # Basic usage
        result = engine.process("open Safari")
        print(result.success)  # True
        print(result.response)  # "Opened Safari"

        # Batch processing
        commands = ["open Chrome", "go to github.com", "scroll down"]
        results = engine.process_batch(commands)

        # Get capabilities
        capabilities = engine.get_capabilities()
        print(capabilities["supported_actions"])
    """

    def __init__(self):
        """Initialize the Aria engine with a command executor."""
        self._executor = CommandExecutor()
        logger.info("AriaEngine initialized")

    def process(self, text: str) -> IntentResult:
        """Process a user request.

        Takes raw text input (possibly from speech recognition), normalizes it,
        parses it into an intent, and executes the appropriate action.

        This method handles commands that can be executed directly without
        AI assistance. For commands that require AI (low confidence or
        complex reasoning), it returns an error result indicating AI is needed.

        Args:
            text: The user's natural language request.

        Returns:
            IntentResult with the outcome of the request:
            - success: True if the command was executed successfully
            - response: Human-readable description of what happened
            - data: Optional additional data from execution
            - error: Error code if execution failed

        Example:
            >>> engine = AriaEngine()
            >>> result = engine.process("click on Finder")
            >>> print(result.success)
            True
            >>> print(result.response)
            "Clicked on Finder"
        """
        if not text or not text.strip():
            logger.warning("Empty input received")
            return IntentResult.error_result(
                error="EMPTY_INPUT",
                response="No command provided"
            )

        logger.debug(f"Processing input: {text}")

        try:
            # Parse the intent
            intent = parse(text)
            logger.debug(f"Parsed intent: action={intent.action.value}, "
                        f"target={intent.target}, confidence={intent.confidence}")

            # Check if AI assistance is required
            if intent.requires_ai:
                logger.info(f"Intent requires AI assistance: {text}")
                return IntentResult(
                    success=False,
                    response="This request requires AI assistance",
                    error="REQUIRES_AI",
                    data={
                        "intent": intent.to_dict(),
                        "raw_text": text
                    }
                )

            # Handle conversation/unknown intents
            if intent.action in [IntentType.CONVERSATION, IntentType.UNKNOWN]:
                logger.info(f"Conversation/unknown intent: {text}")
                return IntentResult(
                    success=False,
                    response="I'm not sure how to help with that. Could you rephrase?",
                    error="CONVERSATION_INTENT",
                    data={
                        "intent": intent.to_dict(),
                        "raw_text": text
                    }
                )

            # Execute the intent
            result = self._executor.execute(intent)
            logger.info(f"Execution result: success={result.success}, "
                       f"response={result.response}")

            return result

        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            return IntentResult.error_result(
                error="PROCESSING_ERROR",
                response=f"Error processing request: {str(e)}"
            )

    def process_with_fallback(self, text: str,
                              ai_handler: Optional[callable] = None) -> IntentResult:
        """Process a request with AI fallback for complex cases.

        First attempts standard processing. If the command requires AI
        assistance (low confidence or complex reasoning), delegates to
        the provided AI handler or returns an error.

        Args:
            text: The user's natural language request.
            ai_handler: Optional callback function for AI processing.
                       Signature: ai_handler(text: str, intent: Intent) -> IntentResult

        Returns:
            IntentResult with the outcome of the request.

        Example:
            def my_ai_handler(text, intent):
                # Custom AI processing logic
                return IntentResult.success_result("AI processed: " + text)

            engine = AriaEngine()
            result = engine.process_with_fallback(
                "analyze the sentiment of this document",
                ai_handler=my_ai_handler
            )
        """
        result = self.process(text)

        # If successful or not an AI-required error, return as-is
        if result.success or result.error not in ["REQUIRES_AI", "CONVERSATION_INTENT"]:
            return result

        # Attempt AI fallback
        if ai_handler:
            try:
                intent_data = result.data.get("intent", {}) if result.data else {}
                intent = Intent(
                    action=IntentType(intent_data.get("action", "unknown")),
                    target=intent_data.get("target"),
                    params=intent_data.get("params", {}),
                    confidence=intent_data.get("confidence", 0.0),
                    raw_text=text,
                    requires_ai=True
                )
                return ai_handler(text, intent)
            except Exception as e:
                logger.error(f"AI handler error: {e}", exc_info=True)
                return IntentResult.error_result(
                    error="AI_HANDLER_ERROR",
                    response=f"AI processing failed: {str(e)}"
                )

        # No AI handler provided
        logger.warning("AI assistance needed but no handler provided")
        return IntentResult(
            success=False,
            response="AI assistance not available for this request",
            error="AI_NOT_IMPLEMENTED",
            data=result.data
        )

    def process_batch(self, commands: List[str],
                      stop_on_error: bool = False) -> List[IntentResult]:
        """Process multiple commands in sequence.

        Executes a list of commands one by one, collecting results.
        Useful for scripted workflows or multi-step operations.

        Args:
            commands: List of natural language commands to execute.
            stop_on_error: If True, stops processing on first error.

        Returns:
            List of IntentResult objects, one for each command.

        Example:
            engine = AriaEngine()
            results = engine.process_batch([
                "open Chrome",
                "go to github.com",
                "scroll down"
            ])
            for result in results:
                print(result.response)
        """
        results = []

        for command in commands:
            result = self.process(command)
            results.append(result)

            if not result.success and stop_on_error:
                logger.warning(f"Stopping batch processing due to error: {result.error}")
                break

        return results

    def parse_only(self, text: str) -> Intent:
        """Parse text into an intent without executing.

        Useful for previewing what action would be taken or for
        implementing confirmation flows.

        Args:
            text: The natural language text to parse.

        Returns:
            Intent object with parsed information.

        Example:
            engine = AriaEngine()
            intent = engine.parse_only("click on Submit button")
            print(f"Would {intent.action.value} on {intent.target}")
            # "Would click on Submit button"
        """
        return parse(text)

    def normalize(self, text: str) -> str:
        """Normalize speech input without parsing or executing.

        Fixes fragmented words and common speech recognition errors.
        Useful for preprocessing text before displaying or logging.

        Args:
            text: Raw speech recognition output.

        Returns:
            Normalized text with corrections applied.

        Example:
            engine = AriaEngine()
            normalized = engine.normalize("clic k on chro me")
            print(normalized)  # "click on chrome"
        """
        return normalize_speech(text)

    def get_capabilities(self) -> Dict[str, Any]:
        """Get information about the engine's capabilities.

        Returns a dictionary describing what the engine can do,
        useful for UI generation or capability discovery.

        Returns:
            Dictionary with:
            - supported_actions: List of supported intent types
            - action_keywords: Keywords that trigger each action
            - version: Engine version string

        Example:
            engine = AriaEngine()
            caps = engine.get_capabilities()
            print(caps["supported_actions"])
            # ["click", "open", "type", "scroll", ...]
        """
        from .intent_parser import get_supported_intents, get_intent_keywords

        actions = {}
        for intent_name in get_supported_intents():
            try:
                intent_type = IntentType(intent_name)
                actions[intent_name] = {
                    "keywords": get_intent_keywords(intent_type),
                    "description": self._get_action_description(intent_type)
                }
            except ValueError:
                pass

        return {
            "supported_actions": get_supported_intents(),
            "actions": actions,
            "version": "3.0.0",
            "name": "AriaEngine",
            "features": [
                "speech_normalization",
                "intent_parsing",
                "accessibility_execution",
                "batch_processing",
            ]
        }

    def _get_action_description(self, action: IntentType) -> str:
        """Get a human-readable description of an action type.

        Args:
            action: The IntentType to describe.

        Returns:
            Description string.
        """
        descriptions = {
            IntentType.CLICK: "Click on UI elements (buttons, icons, links)",
            IntentType.OPEN: "Open applications, files, or URLs",
            IntentType.TYPE: "Type text into focused fields",
            IntentType.SCROLL: "Scroll up, down, or to specific positions",
            IntentType.CLOSE: "Close windows, tabs, or applications",
            IntentType.TAB: "Manage browser tabs (new, close, switch)",
            IntentType.KEYBOARD: "Execute keyboard shortcuts",
            IntentType.NAVIGATE: "Navigate to URLs in the browser",
            IntentType.MEMORY: "Store and retrieve information from memory",
            IntentType.CONVERSATION: "General conversation (requires AI)",
            IntentType.UNKNOWN: "Unrecognized command",
        }
        return descriptions.get(action, "Unknown action")


# ============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ============================================================================

# Singleton engine instance
_engine: Optional[AriaEngine] = None


def get_engine() -> AriaEngine:
    """Get the shared AriaEngine instance.

    Returns:
        Singleton AriaEngine instance.

    Example:
        from aria.core.aria_engine import get_engine

        engine = get_engine()
        result = engine.process("click on Chrome")
    """
    global _engine
    if _engine is None:
        _engine = AriaEngine()
    return _engine


def process(text: str) -> IntentResult:
    """Process a command using the default engine.

    Convenience function for simple one-off processing.

    Args:
        text: Natural language command.

    Returns:
        IntentResult with execution outcome.

    Example:
        from aria.core.aria_engine import process

        result = process("open Safari")
        print(result.response)
    """
    return get_engine().process(text)


def process_with_fallback(text: str, ai_handler: Optional[callable] = None) -> IntentResult:
    """Process a command with AI fallback using the default engine.

    Args:
        text: Natural language command.
        ai_handler: Optional AI processing callback.

    Returns:
        IntentResult with execution outcome.
    """
    return get_engine().process_with_fallback(text, ai_handler)
