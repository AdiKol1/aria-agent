"""Base classes for intent handling.

This module provides the foundational data structures for the intent parsing system.
It defines the types of intents that can be recognized, the structure of parsed intents,
and the results from executing those intents.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class IntentType(Enum):
    """Enumeration of all supported intent types.

    Each intent type corresponds to a category of user actions that can be
    recognized and handled by the system.
    """
    CLICK = "click"           # Mouse click actions
    OPEN = "open"             # Opening apps, files, URLs
    TYPE = "type"             # Typing text
    SCROLL = "scroll"         # Scrolling actions
    CLOSE = "close"           # Closing apps, windows, tabs
    TAB = "tab"               # Tab management (new, close, switch)
    KEYBOARD = "keyboard"     # Keyboard shortcuts (copy, paste, etc.)
    NAVIGATE = "navigate"     # URL/page navigation
    MEMORY = "memory"         # Memory operations (remember, recall)
    CONVERSATION = "conversation"  # Falls through to AI for response
    UNKNOWN = "unknown"       # Unrecognized intent


@dataclass
class Intent:
    """Represents a parsed user intent.

    This class encapsulates all information extracted from a user's natural
    language input, including the action to take, the target of that action,
    any additional parameters, and metadata about the parsing confidence.

    Attributes:
        action: The type of intent identified (from IntentType enum).
        target: The target of the action (e.g., app name, button, URL).
        params: Additional parameters needed to execute the intent.
        confidence: Score from 0.0 to 1.0 indicating parsing confidence.
        raw_text: The original user input before processing.
        requires_ai: Whether this intent needs AI assistance to execute.

    Example:
        >>> intent = Intent(
        ...     action=IntentType.CLICK,
        ...     target="Chrome",
        ...     params={"location": "dock"},
        ...     confidence=0.95,
        ...     raw_text="click on Chrome in the dock"
        ... )
    """
    action: IntentType
    target: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    raw_text: str = ""
    requires_ai: bool = False

    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Check if the intent has high confidence.

        Args:
            threshold: Minimum confidence score to be considered high.

        Returns:
            True if confidence meets or exceeds the threshold.
        """
        return self.confidence >= threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert intent to dictionary representation.

        Returns:
            Dictionary containing all intent attributes.
        """
        return {
            "action": self.action.value,
            "target": self.target,
            "params": self.params,
            "confidence": self.confidence,
            "raw_text": self.raw_text,
            "requires_ai": self.requires_ai
        }


@dataclass
class IntentResult:
    """Result from executing an intent.

    This class represents the outcome of attempting to execute a parsed intent,
    including whether it succeeded, any response message, and additional data
    or error information.

    Attributes:
        success: Whether the intent execution succeeded.
        response: Human-readable response message.
        data: Optional additional data from the execution.
        error: Error message if execution failed.

    Example:
        >>> result = IntentResult(
        ...     success=True,
        ...     response="Opened Chrome successfully",
        ...     data={"app_pid": 12345}
        ... )
    """
    success: bool
    response: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @classmethod
    def success_result(cls, response: str, data: Optional[Dict[str, Any]] = None) -> "IntentResult":
        """Create a successful result.

        Args:
            response: The success message.
            data: Optional additional data.

        Returns:
            An IntentResult with success=True.
        """
        return cls(success=True, response=response, data=data)

    @classmethod
    def error_result(cls, error: str, response: Optional[str] = None) -> "IntentResult":
        """Create an error result.

        Args:
            error: The error message.
            response: Optional human-readable response.

        Returns:
            An IntentResult with success=False.
        """
        return cls(
            success=False,
            response=response or f"Error: {error}",
            error=error
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation.

        Returns:
            Dictionary containing all result attributes.
        """
        return {
            "success": self.success,
            "response": self.response,
            "data": self.data,
            "error": self.error
        }


class IntentHandler:
    """Base class for intent handlers.

    All specific intent handlers should inherit from this class and implement
    the required methods. This provides a consistent interface for the intent
    processing pipeline.
    """

    # Subclasses should define their trigger patterns
    PATTERNS: List[str] = []

    def can_handle(self, text: str) -> float:
        """Determine if this handler can process the given text.

        Args:
            text: The user input text to evaluate.

        Returns:
            Confidence score from 0.0 to 1.0 indicating how well this
            handler matches the input.
        """
        raise NotImplementedError("Subclasses must implement can_handle()")

    def extract_params(self, text: str) -> Dict[str, Any]:
        """Extract parameters from the input text.

        Args:
            text: The user input text to parse.

        Returns:
            Dictionary of extracted parameters.
        """
        raise NotImplementedError("Subclasses must implement extract_params()")

    def handle(self, intent: Intent) -> IntentResult:
        """Execute the intent.

        Args:
            intent: The parsed intent to execute.

        Returns:
            The result of executing the intent.
        """
        raise NotImplementedError("Subclasses must implement handle()")
