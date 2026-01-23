"""Intent handlers for the Aria agent.

This package contains all intent-related classes including:
- Base classes for intent representation and handling
- Specific handlers for different intent types (click, open, type, etc.)

Example usage:
    from aria.intents import Intent, IntentType, IntentResult
    from aria.intents import ClickIntentHandler, OpenIntentHandler

    # Create an intent
    intent = Intent(
        action=IntentType.CLICK,
        target="Chrome",
        confidence=0.95
    )

    # Use a handler
    handler = ClickIntentHandler()
    if handler.can_handle("click on Chrome"):
        params = handler.extract_params("click on Chrome")
"""

from .base import (
    IntentType,
    Intent,
    IntentResult,
    IntentHandler,
)

from .click import ClickIntentHandler
from .open import OpenIntentHandler
from .type import TypeIntentHandler
from .scroll import ScrollIntentHandler
from .keyboard import KeyboardIntentHandler
from .tab import TabIntentHandler

__all__ = [
    # Base classes
    "IntentType",
    "Intent",
    "IntentResult",
    "IntentHandler",
    # Handlers
    "ClickIntentHandler",
    "OpenIntentHandler",
    "TypeIntentHandler",
    "ScrollIntentHandler",
    "KeyboardIntentHandler",
    "TabIntentHandler",
]


# Registry of all intent handlers
INTENT_HANDLERS = {
    IntentType.CLICK: ClickIntentHandler,
    IntentType.OPEN: OpenIntentHandler,
    IntentType.TYPE: TypeIntentHandler,
    IntentType.SCROLL: ScrollIntentHandler,
    IntentType.KEYBOARD: KeyboardIntentHandler,
    IntentType.TAB: TabIntentHandler,
}


def get_handler_for_intent(intent_type: IntentType) -> IntentHandler:
    """Get the appropriate handler for an intent type.

    Args:
        intent_type: The type of intent to handle.

    Returns:
        An instance of the appropriate handler.

    Raises:
        ValueError: If no handler exists for the intent type.
    """
    handler_class = INTENT_HANDLERS.get(intent_type)
    if handler_class is None:
        raise ValueError(f"No handler registered for intent type: {intent_type}")
    return handler_class()


def get_all_handlers() -> list:
    """Get instances of all registered handlers.

    Returns:
        List of handler instances.
    """
    return [handler_class() for handler_class in INTENT_HANDLERS.values()]
