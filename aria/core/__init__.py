"""Core components for the Aria agent.

This package contains the core parsing, execution, and processing logic for the
intent-based natural language interface.

Main Components:
- IntentParser: Converts natural language to structured Intent objects
- CommandExecutor: Executes intents using the Accessibility API
- AriaEngine: High-level processing pipeline combining parsing and execution
- AIHandler: Claude integration for complex cases that require AI reasoning
- ConversationHandler: Handles pure conversational requests
- VoiceBridge: Bridges voice transcription to AriaEngine for reliable action execution

Example usage:
    from aria.core import AriaEngine

    # Create the engine
    engine = AriaEngine()

    # Process a command
    result = engine.process("click on Chrome")
    print(result.success)  # True
    print(result.response)  # "Clicked on Google Chrome"

    # Or use lower-level components
    from aria.core import parse, CommandExecutor

    intent = parse("click on Chrome")
    executor = CommandExecutor()
    result = executor.execute(intent)

    # Normalize fragmented speech
    from aria.core import normalize_speech
    normalized = normalize_speech("clic k on chro me")
    print(normalized)  # "click on chrome"

    # Use AI fallback for complex cases
    from aria.core import AriaEngine, AIHandler

    engine = AriaEngine()
    ai = AIHandler()

    result = engine.process_with_fallback(
        "help me organize my desktop",
        ai_handler=ai.handle
    )
"""

from .intent_parser import (
    # Main parsing function
    parse,
    # Cleaning and normalization
    clean_voice_input,
    normalize_speech,
    # Target extraction
    extract_target,
    # Confidence scoring
    calculate_confidence,
    # Utility functions
    get_supported_intents,
    get_intent_keywords,
    is_action_keyword,
)

from .executor import (
    CommandExecutor,
    execute_intent,
)

from .aria_engine import (
    AriaEngine,
    get_engine,
    process,
    process_with_fallback,
)

from .ai_handler import (
    AIHandler,
    ConversationHandler,
    get_ai_handler,
    get_conversation_handler,
    create_ai_callback,
)

from .voice_bridge import (
    VoiceBridge,
    get_voice_bridge,
    process_voice,
    should_intercept_voice,
)

__all__ = [
    # High-level API
    "AriaEngine",
    "get_engine",
    "process",
    "process_with_fallback",
    # AI Integration
    "AIHandler",
    "ConversationHandler",
    "get_ai_handler",
    "get_conversation_handler",
    "create_ai_callback",
    # Executor
    "CommandExecutor",
    "execute_intent",
    # Parser functions
    "parse",
    "clean_voice_input",
    "normalize_speech",
    "extract_target",
    "calculate_confidence",
    # Utilities
    "get_supported_intents",
    "get_intent_keywords",
    "is_action_keyword",
    # Voice Bridge
    "VoiceBridge",
    "get_voice_bridge",
    "process_voice",
    "should_intercept_voice",
]
