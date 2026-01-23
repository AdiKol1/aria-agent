"""AI Handler - Claude integration for complex cases.

This module provides an AI handler that can be passed to AriaEngine.process_with_fallback()
to handle requests that require AI reasoning or multi-step planning.

The AIHandler is designed to work seamlessly with the intent-based processing pipeline,
providing intelligent fallback behavior for:
1. Low-confidence intent parsing
2. Conversational requests
3. Complex multi-step tasks
4. Context-dependent actions

Example:
    from aria.core import AriaEngine
    from aria.core.ai_handler import AIHandler

    engine = AriaEngine()
    ai = AIHandler()

    # Process with AI fallback
    result = engine.process_with_fallback(
        "help me organize my desktop",
        ai_handler=ai.handle
    )
    print(result.response)
"""
import os
import logging
from typing import Optional, Dict, Any, List, Callable

from anthropic import Anthropic

from ..intents.base import Intent, IntentType, IntentResult
from ..config import ANTHROPIC_API_KEY, CLAUDE_MODEL

logger = logging.getLogger(__name__)


# Default model for AI handler - use config value
DEFAULT_MODEL = CLAUDE_MODEL


class AIHandler:
    """Handles complex requests using Claude.

    This handler is designed to be passed to AriaEngine.process_with_fallback()
    for cases where the intent parser cannot confidently determine the action.

    The handler uses Claude with tool definitions to:
    - Understand ambiguous or context-dependent requests
    - Plan and execute multi-step tasks
    - Provide intelligent responses when actions aren't appropriate

    Attributes:
        _client: Anthropic API client instance.
        _model: Claude model identifier to use.
        _tools: Tool definitions for Claude.
        _tool_handlers: Mapping of tool names to handler functions.

    Example:
        from aria.core import AriaEngine
        from aria.core.ai_handler import AIHandler

        engine = AriaEngine()
        ai = AIHandler()

        result = engine.process_with_fallback(
            "help me organize my desktop",
            ai_handler=ai.handle
        )
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None
    ):
        """Initialize the AI handler.

        Args:
            model: Claude model identifier. Defaults to the configured CLAUDE_MODEL.
            api_key: Anthropic API key. If not provided, uses ANTHROPIC_API_KEY from config.
        """
        # Use provided API key, or fall back to config
        effective_api_key = api_key or ANTHROPIC_API_KEY
        if not effective_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it in your .env file or pass api_key parameter."
            )

        self._client = Anthropic(api_key=effective_api_key)
        self._model = model
        self._tools = self._build_tools()
        self._tool_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()

        logger.info(f"AIHandler initialized with model: {model}")

    def _build_tools(self) -> List[Dict[str, Any]]:
        """Build tool definitions for Claude.

        Returns:
            List of tool definitions in Anthropic's tool format.
        """
        return [
            {
                "name": "click",
                "description": "Click on a UI element by name or description. The element will be found using accessibility APIs.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "The name or description of the element to click (e.g., 'Chrome', 'Submit button', 'Settings icon')"
                        },
                        "location": {
                            "type": "string",
                            "description": "Optional location hint (e.g., 'dock', 'menu bar', 'toolbar')"
                        }
                    },
                    "required": ["target"]
                }
            },
            {
                "name": "open_app",
                "description": "Open an application by name",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "app_name": {
                            "type": "string",
                            "description": "The name of the application to open (e.g., 'Safari', 'VS Code', 'Finder')"
                        }
                    },
                    "required": ["app_name"]
                }
            },
            {
                "name": "open_url",
                "description": "Open a URL in the default browser",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to open (e.g., 'https://github.com')"
                        }
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "type_text",
                "description": "Type text into the currently focused field",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to type"
                        }
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "press_key",
                "description": "Press a key or key combination",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "The key to press (e.g., 'enter', 'escape', 'tab', 'space')"
                        }
                    },
                    "required": ["key"]
                }
            },
            {
                "name": "hotkey",
                "description": "Execute a keyboard shortcut",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of keys for the shortcut (e.g., ['command', 'c'] for copy)"
                        }
                    },
                    "required": ["keys"]
                }
            },
            {
                "name": "scroll",
                "description": "Scroll up or down",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["up", "down"],
                            "description": "Direction to scroll"
                        },
                        "amount": {
                            "type": "integer",
                            "description": "Amount to scroll (default 300 for one 'page')"
                        }
                    },
                    "required": ["direction"]
                }
            },
            {
                "name": "close_window",
                "description": "Close the current window or tab",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "What to close ('window', 'tab', or app name)"
                        }
                    },
                    "required": ["target"]
                }
            },
            {
                "name": "new_tab",
                "description": "Open a new browser tab",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Optional URL to open in the new tab"
                        }
                    }
                }
            },
            {
                "name": "switch_tab",
                "description": "Switch to a different browser tab",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["next", "previous"],
                            "description": "Direction to switch tabs"
                        },
                        "tab_number": {
                            "type": "integer",
                            "description": "Specific tab number to switch to (1-indexed)"
                        }
                    }
                }
            },
            {
                "name": "respond",
                "description": "Respond to the user with a message (no action needed)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The response message to the user"
                        }
                    },
                    "required": ["message"]
                }
            }
        ]

    def _register_default_handlers(self):
        """Register default tool handlers that convert tool calls to IntentResults."""
        # Import here to avoid circular imports
        from ..core.executor import CommandExecutor

        try:
            self._executor = CommandExecutor()
        except Exception as e:
            logger.warning(f"Could not initialize CommandExecutor: {e}")
            self._executor = None

        # Register handlers for each tool
        self._tool_handlers = {
            "click": self._handle_click,
            "open_app": self._handle_open_app,
            "open_url": self._handle_open_url,
            "type_text": self._handle_type_text,
            "press_key": self._handle_press_key,
            "hotkey": self._handle_hotkey,
            "scroll": self._handle_scroll,
            "close_window": self._handle_close,
            "new_tab": self._handle_new_tab,
            "switch_tab": self._handle_switch_tab,
            "respond": self._handle_respond,
        }

    def register_tool_handler(
        self,
        tool_name: str,
        handler: Callable[[Dict[str, Any]], IntentResult]
    ):
        """Register a custom handler for a tool.

        Args:
            tool_name: Name of the tool to handle.
            handler: Function that takes tool input and returns IntentResult.
        """
        self._tool_handlers[tool_name] = handler
        logger.debug(f"Registered custom handler for tool: {tool_name}")

    def handle(self, text: str, intent: Intent) -> IntentResult:
        """Handle a request using Claude.

        This method serves as the ai_handler callback for process_with_fallback().

        Args:
            text: The original user request.
            intent: The partially parsed intent (may have low confidence).

        Returns:
            IntentResult from AI processing.
        """
        try:
            logger.info(f"AI handling request: {text[:50]}...")

            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=self._get_system_prompt(),
                messages=[{"role": "user", "content": text}],
                tools=self._tools
            )

            return self._process_response(response, text)

        except Exception as e:
            logger.error(f"AI handler error: {e}", exc_info=True)
            return IntentResult.error_result(
                error="AI_ERROR",
                response=f"AI processing failed: {str(e)}"
            )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for Claude.

        Returns:
            System prompt string.
        """
        return """You are Aria, a helpful AI assistant that controls a Mac computer.

When the user asks you to do something:
1. Use the available tools to accomplish the task
2. Be direct - execute actions rather than just describing them
3. If you need to perform multiple steps, do them in order
4. If the user is having a conversation or asking a question, use the "respond" tool

Available actions through tools:
- Click on UI elements (buttons, icons, menus)
- Open applications
- Open URLs in browser
- Type text
- Use keyboard shortcuts
- Scroll up/down
- Close windows/tabs
- Manage browser tabs (new, close, switch)

Guidelines:
- For actions: Use the appropriate tool
- For questions/conversation: Use the "respond" tool with your answer
- Be concise but helpful
- If you're unsure what the user wants, ask for clarification using "respond"

Always try to help the user accomplish their goal."""

    def _process_response(self, response: Any, original_text: str) -> IntentResult:
        """Process Claude's response and execute any tool calls.

        Args:
            response: Claude API response object.
            original_text: The original user request.

        Returns:
            IntentResult from processing.
        """
        # Check stop reason
        if response.stop_reason == "tool_use":
            # Find tool use blocks
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input

                    logger.info(f"Executing AI tool: {tool_name}")

                    # Execute the tool
                    handler = self._tool_handlers.get(tool_name)
                    if handler:
                        try:
                            return handler(tool_input)
                        except Exception as e:
                            logger.error(f"Tool execution error: {e}")
                            return IntentResult.error_result(
                                error="TOOL_EXECUTION_ERROR",
                                response=f"Failed to execute {tool_name}: {str(e)}"
                            )
                    else:
                        logger.warning(f"No handler for tool: {tool_name}")
                        return IntentResult.error_result(
                            error="UNKNOWN_TOOL",
                            response=f"Unknown tool: {tool_name}"
                        )

        # No tool use - extract text response
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content += block.text

        if text_content:
            return IntentResult.success_result(
                response=text_content,
                data={"source": "ai_response"}
            )

        return IntentResult.error_result(
            error="NO_RESPONSE",
            response="AI did not provide a response"
        )

    # =========================================================================
    # Tool Handlers
    # =========================================================================

    def _handle_click(self, input_data: Dict[str, Any]) -> IntentResult:
        """Handle click tool."""
        target = input_data.get("target", "")
        location = input_data.get("location")

        if self._executor:
            intent = Intent(
                action=IntentType.CLICK,
                target=target,
                params={"location": location} if location else {},
                confidence=0.9,
                raw_text=f"click on {target}"
            )
            return self._executor.execute(intent)

        return IntentResult.success_result(
            response=f"Would click on {target}",
            data={"action": "click", "target": target}
        )

    def _handle_open_app(self, input_data: Dict[str, Any]) -> IntentResult:
        """Handle open app tool."""
        app_name = input_data.get("app_name", "")

        if self._executor:
            intent = Intent(
                action=IntentType.OPEN,
                target=app_name,
                params={"type": "app"},
                confidence=0.9,
                raw_text=f"open {app_name}"
            )
            return self._executor.execute(intent)

        return IntentResult.success_result(
            response=f"Would open {app_name}",
            data={"action": "open_app", "app": app_name}
        )

    def _handle_open_url(self, input_data: Dict[str, Any]) -> IntentResult:
        """Handle open URL tool."""
        url = input_data.get("url", "")

        if self._executor:
            intent = Intent(
                action=IntentType.OPEN,
                target=url,
                params={"type": "url"},
                confidence=0.9,
                raw_text=f"open {url}"
            )
            return self._executor.execute(intent)

        return IntentResult.success_result(
            response=f"Would open {url}",
            data={"action": "open_url", "url": url}
        )

    def _handle_type_text(self, input_data: Dict[str, Any]) -> IntentResult:
        """Handle type text tool."""
        text = input_data.get("text", "")

        if self._executor:
            intent = Intent(
                action=IntentType.TYPE,
                target=text,
                confidence=0.9,
                raw_text=f"type {text}"
            )
            return self._executor.execute(intent)

        return IntentResult.success_result(
            response=f"Would type: {text[:30]}...",
            data={"action": "type", "text": text}
        )

    def _handle_press_key(self, input_data: Dict[str, Any]) -> IntentResult:
        """Handle press key tool."""
        key = input_data.get("key", "")

        if self._executor:
            intent = Intent(
                action=IntentType.KEYBOARD,
                target=key,
                params={"type": "key"},
                confidence=0.9,
                raw_text=f"press {key}"
            )
            return self._executor.execute(intent)

        return IntentResult.success_result(
            response=f"Would press {key}",
            data={"action": "press", "key": key}
        )

    def _handle_hotkey(self, input_data: Dict[str, Any]) -> IntentResult:
        """Handle hotkey tool."""
        keys = input_data.get("keys", [])

        if self._executor:
            intent = Intent(
                action=IntentType.KEYBOARD,
                target="+".join(keys),
                params={"type": "hotkey", "keys": keys},
                confidence=0.9,
                raw_text=f"press {'+'.join(keys)}"
            )
            return self._executor.execute(intent)

        return IntentResult.success_result(
            response=f"Would press {'+'.join(keys)}",
            data={"action": "hotkey", "keys": keys}
        )

    def _handle_scroll(self, input_data: Dict[str, Any]) -> IntentResult:
        """Handle scroll tool."""
        direction = input_data.get("direction", "down")
        amount = input_data.get("amount", 300)

        # Convert direction to signed amount
        scroll_amount = amount if direction == "up" else -amount

        if self._executor:
            intent = Intent(
                action=IntentType.SCROLL,
                target=direction,
                params={"amount": scroll_amount},
                confidence=0.9,
                raw_text=f"scroll {direction}"
            )
            return self._executor.execute(intent)

        return IntentResult.success_result(
            response=f"Would scroll {direction}",
            data={"action": "scroll", "direction": direction, "amount": scroll_amount}
        )

    def _handle_close(self, input_data: Dict[str, Any]) -> IntentResult:
        """Handle close window/tab tool."""
        target = input_data.get("target", "window")

        if self._executor:
            intent = Intent(
                action=IntentType.CLOSE,
                target=target,
                confidence=0.9,
                raw_text=f"close {target}"
            )
            return self._executor.execute(intent)

        return IntentResult.success_result(
            response=f"Would close {target}",
            data={"action": "close", "target": target}
        )

    def _handle_new_tab(self, input_data: Dict[str, Any]) -> IntentResult:
        """Handle new tab tool."""
        url = input_data.get("url")

        if self._executor:
            intent = Intent(
                action=IntentType.TAB,
                target="new",
                params={"url": url} if url else {},
                confidence=0.9,
                raw_text="new tab" if not url else f"new tab {url}"
            )
            return self._executor.execute(intent)

        return IntentResult.success_result(
            response=f"Would open new tab" + (f" to {url}" if url else ""),
            data={"action": "new_tab", "url": url}
        )

    def _handle_switch_tab(self, input_data: Dict[str, Any]) -> IntentResult:
        """Handle switch tab tool."""
        direction = input_data.get("direction")
        tab_number = input_data.get("tab_number")

        if self._executor:
            intent = Intent(
                action=IntentType.TAB,
                target="switch",
                params={
                    "direction": direction,
                    "tab_number": tab_number
                },
                confidence=0.9,
                raw_text=f"switch to {direction or tab_number} tab"
            )
            return self._executor.execute(intent)

        return IntentResult.success_result(
            response=f"Would switch tab",
            data={"action": "switch_tab", "direction": direction, "tab_number": tab_number}
        )

    def _handle_respond(self, input_data: Dict[str, Any]) -> IntentResult:
        """Handle respond tool (for conversational responses)."""
        message = input_data.get("message", "")

        return IntentResult.success_result(
            response=message,
            data={"source": "ai_conversation"}
        )


class ConversationHandler:
    """Handles conversational requests (no action needed).

    This handler is for pure conversation where no computer control actions
    are required. It provides direct, helpful responses to user questions.

    Attributes:
        _client: Anthropic API client instance.
        _model: Claude model identifier to use.

    Example:
        from aria.core.ai_handler import ConversationHandler

        handler = ConversationHandler()
        response = handler.respond("What's the weather like?")
        print(response)  # Conversational response from Claude
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None
    ):
        """Initialize the conversation handler.

        Args:
            model: Claude model identifier. Defaults to the configured CLAUDE_MODEL.
            api_key: Anthropic API key. If not provided, uses ANTHROPIC_API_KEY from config.
        """
        # Use provided API key, or fall back to config
        effective_api_key = api_key or ANTHROPIC_API_KEY
        if not effective_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it in your .env file or pass api_key parameter."
            )

        self._client = Anthropic(api_key=effective_api_key)
        self._model = model
        logger.info(f"ConversationHandler initialized with model: {model}")

    def respond(self, text: str, context: Optional[str] = None) -> str:
        """Generate a conversational response.

        Args:
            text: The user's message.
            context: Optional additional context (e.g., from memory).

        Returns:
            Claude's response as a string.
        """
        try:
            messages = [{"role": "user", "content": text}]

            system_prompt = """You are Aria, a friendly and helpful AI assistant.
You are having a conversation with the user - no computer actions are needed.
Be natural, helpful, and concise. Match your response length to the complexity
of the question."""

            if context:
                system_prompt += f"\n\nContext about the user:\n{context}"

            response = self._client.messages.create(
                model=self._model,
                max_tokens=500,
                system=system_prompt,
                messages=messages
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Conversation handler error: {e}", exc_info=True)
            return f"I apologize, but I encountered an error: {str(e)}"

    def respond_with_memory(
        self,
        text: str,
        memory_facts: Optional[List[str]] = None
    ) -> str:
        """Generate a response using memory context.

        Args:
            text: The user's message.
            memory_facts: List of relevant facts from memory.

        Returns:
            Claude's response as a string.
        """
        context = None
        if memory_facts:
            context = "Known information:\n" + "\n".join(f"- {fact}" for fact in memory_facts)

        return self.respond(text, context)


# ============================================================================
# Factory Functions
# ============================================================================

_ai_handler: Optional[AIHandler] = None
_conversation_handler: Optional[ConversationHandler] = None


def get_ai_handler(model: Optional[str] = None) -> AIHandler:
    """Get a shared AIHandler instance.

    Args:
        model: Optional model override. Uses default if not specified.

    Returns:
        AIHandler singleton instance.
    """
    global _ai_handler
    if _ai_handler is None:
        _ai_handler = AIHandler(model=model or DEFAULT_MODEL)
    return _ai_handler


def get_conversation_handler(model: Optional[str] = None) -> ConversationHandler:
    """Get a shared ConversationHandler instance.

    Args:
        model: Optional model override. Uses default if not specified.

    Returns:
        ConversationHandler singleton instance.
    """
    global _conversation_handler
    if _conversation_handler is None:
        _conversation_handler = ConversationHandler(model=model or DEFAULT_MODEL)
    return _conversation_handler


def create_ai_callback(
    model: Optional[str] = None
) -> Callable[[str, Intent], IntentResult]:
    """Create an AI callback for use with process_with_fallback.

    This is a convenience function that returns a properly configured
    callback function for AriaEngine.process_with_fallback().

    Args:
        model: Optional model override.

    Returns:
        Callback function with signature (text, intent) -> IntentResult

    Example:
        from aria.core import AriaEngine
        from aria.core.ai_handler import create_ai_callback

        engine = AriaEngine()
        result = engine.process_with_fallback(
            "help me with something complex",
            ai_handler=create_ai_callback()
        )
    """
    handler = get_ai_handler(model)
    return handler.handle
