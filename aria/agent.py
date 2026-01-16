"""
Aria Agent Brain v2.0

The core reasoning engine with intelligent planning, learning, and self-correction.

NOTE: anthropic import is deferred via lazy_anthropic module to avoid slow startup (~40s).
"""

import json
import time
import hashlib
import asyncio
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING

from .config import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MODEL_FAST, CLAUDE_MAX_TOKENS,
    ACTION_DELAY_MS, MCP_SERVERS, MCP_AUTO_START
)
from .vision import get_claude_vision, get_screen_capture
from .control import get_control
from .memory import get_memory
from .claude_bridge import get_claude_bridge
from .intent import get_intent_engine, Intent
from .planner import get_planner, Plan, PlanStep
from .learning import get_learning_engine
from .clarification import get_clarification_engine, Clarification
from .skills import (
    get_registry, get_loader, get_hooks,
    SkillContext, SkillResult, HookEvent, create_default_hooks
)
from .mcp_client import get_mcp_client, MCPClient
from .gestures import (
    get_gesture_controller, GestureController, GestureEvent,
    Gesture, GestureAction
)
from .reasoning import get_reasoner, RequestType, MultiModelReasoner
from .lazy_anthropic import get_client as get_anthropic_client
from .tool_search import get_tool_search_manager, TOOL_SEARCH_BETA, ToolSearchManager


ARIA_SYSTEM_PROMPT = """You are Aria, a highly intelligent AI assistant with the ability to control a Mac computer.

## YOUR IDENTITY
You are NOT just a macro executor. You are a brilliant, thoughtful AI assistant who:
- Engages in natural, intelligent conversation
- Answers questions thoroughly and helpfully
- Provides advice, explanations, and insights
- AND can also take actions on the computer when asked

## CONVERSATION MODE (DEFAULT)
For questions, chat, or discussions, respond naturally:
- Give thoughtful, complete answers
- Share relevant knowledge and insights
- Be warm but not overly casual
- Match your response length to the question's complexity
- You have memory - use it to personalize responses

## ACTION MODE (Only when explicitly requested)
ONLY use actions when the user explicitly asks you to DO something on their computer:
- "Click on X" → Action needed
- "Open Safari" → Action needed
- "Scroll down" → Action needed
- "What is X?" → NO action - just answer the question!
- "Tell me about X" → NO action - just have a conversation!

When action IS needed, include JSON:
```json
{"action": "click", "x": 500, "y": 300}
```

Available actions:
- click: {"action": "click", "x": X, "y": Y}
- double_click: {"action": "double_click", "x": X, "y": Y}
- type: {"action": "type", "text": "text here"}
- press: {"action": "press", "key": "enter"}
- hotkey: {"action": "hotkey", "keys": ["command", "w"]}
- scroll: {"action": "scroll", "amount": -3}
- open_app: {"action": "open_app", "app": "Safari"}
- open_url: {"action": "open_url", "url": "https://example.com"}
- wait: {"action": "wait", "seconds": 2}

## CRITICAL DISTINCTION
- User asks "What can you do?" → ANSWER with words, describe your capabilities
- User says "Open Safari" → USE ACTION to actually open Safari
- User asks "What's on my screen?" → DESCRIBE what you see
- User says "Click on the browser" → USE ACTION to click

## MEMORY
You have long-term memory in [MEMORY CONTEXT] blocks. Use it to:
- Remember user preferences
- Resolve references ("that thing we discussed")
- Personalize your responses

Be intelligent. Be helpful. Have real conversations. Only act when asked to act.
"""

# Action-focused prompt for when we've confirmed the user wants an action
ARIA_ACTION_PROMPT = """You are Aria, executing a computer action for the user.

The user has requested you take an action. Analyze the screen and execute.

## RULES
1. Keep spoken responses to 1 SHORT sentence
2. Include action JSON for what needs to be done
3. After each action, verify it worked
4. For multi-step tasks, do ONE action at a time

## Actions
```json
{"action": "click", "x": 500, "y": 300}
```

Available: click, double_click, type, press, hotkey, scroll, open_app, open_url, wait

## COMPLETING TASKS
For SIMPLE tasks, mark done immediately:
```json
{"action": "scroll", "amount": -300, "done": true}
```

For MULTI-STEP tasks, continue until goal reached, then:
```json
{"done": true}
```

Focus on executing the action efficiently.
"""


class AriaAgent:
    """The Aria agent brain v2.0 - with intelligent planning and learning."""

    def __init__(self):
        # Use shared lazy-loaded anthropic client
        self.client = get_anthropic_client(ANTHROPIC_API_KEY)
        self.vision = get_claude_vision()
        self.screen = get_screen_capture()
        self.control = get_control()
        self.memory = get_memory()
        self.claude_bridge = get_claude_bridge()
        self.conversation_history: List[Dict[str, Any]] = []

        # New v2.0 components
        self.intent_engine = get_intent_engine()
        self.planner = get_planner()
        self.learning = get_learning_engine()
        self.clarification = get_clarification_engine()

        # Multi-model reasoning for intelligent responses
        # Pass our existing anthropic client to avoid double-importing
        self.reasoner = get_reasoner(claude_client=self.client)

        # Task tracking
        self.current_task_id: Optional[str] = None
        self.failure_count: int = 0

        # Skills system
        self.skill_registry = get_registry()
        self.skill_loader = get_loader()
        self.hooks = get_hooks()

        # MCP client for external tool servers
        self.mcp_client = get_mcp_client()
        self._initialize_mcp()

        # Load skills and set up hooks
        self._initialize_skills()

        # Gesture recognition for hands-free control
        self.gesture_controller: Optional[GestureController] = None
        self.gesture_enabled = False
        self._pending_gesture_confirmation: Optional[Dict[str, Any]] = None

        # Tool Search - efficient tool management for Claude
        self.tool_search_manager = get_tool_search_manager()
        self._register_tool_handlers()

        # For coordinate scaling (Retina displays)
        import pyautogui
        self.logical_screen_size = pyautogui.size()
        self.image_size_for_claude = (1920, 1242)

        print(f"=== Aria v2.0 Initialized ===")
        print(f"Screen size (logical): {self.logical_screen_size}")
        print(f"Memory loaded: {len(self.memory.get_all_facts())} facts")
        print(f"Learning data: {len(self.learning.outcomes)} past outcomes")
        print(f"Skills loaded: {self.skill_registry.count()} skills")
        print(f"MCP servers registered: {len(self.mcp_client.servers)}")
        print(f"Tool Search: {self.tool_search_manager.total_tool_count} tools ({self.tool_search_manager.deferred_tool_count} deferred)")
        print(f"Gesture recognition: available (call enable_gestures() to start)")
        print(f"Claude Code bridge ready")

        # Trigger session start hooks
        hook_result = self.hooks.trigger_session_start()
        if hook_result.context_injection:
            print(f"Session context: {hook_result.context_injection[:50]}...")

    def _initialize_mcp(self):
        """Initialize MCP servers from configuration."""
        # Register servers from config (they may already be registered by the singleton)
        for name, config in MCP_SERVERS.items():
            if config.get("enabled", True) and name not in self.mcp_client.servers:
                self.mcp_client.register_server(
                    name=name,
                    command=config["command"],
                    args=config["args"],
                    env=config.get("env")
                )

        # Auto-start servers if configured
        if MCP_AUTO_START:
            for name in self.mcp_client.servers:
                try:
                    self.mcp_client.start_server(name)
                except Exception as e:
                    print(f"Warning: Failed to auto-start MCP server {name}: {e}")

        # Register MCP tools with Tool Search manager (as deferred tools)
        self._register_mcp_tools_for_search()

    def _register_mcp_tools_for_search(self):
        """Register MCP server tools with the Tool Search manager."""
        for server_name, server in self.mcp_client.servers.items():
            if server.initialized and server.tools:
                # Register tools with defer_loading for efficient loading
                self.tool_search_manager.register_mcp_tools(server.tools, server_name)

                # Also register handlers for each MCP tool
                for tool in server.tools:
                    tool_full_name = f"{server_name}_{tool['name']}"
                    self.tool_search_manager.register_tool_handler(
                        tool_full_name,
                        lambda args, sn=server_name, tn=tool['name']: self._tool_mcp_call(sn, tn, args)
                    )

    def _tool_mcp_call(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Generic handler for MCP tool calls."""
        result = self.mcp_client.call_tool(server_name, tool_name, arguments)
        if "error" in result:
            return {"success": False, "error": result["error"]}
        return {"success": True, "result": result}

    def _initialize_skills(self):
        """Initialize the skills system."""
        from pathlib import Path

        # Create default hooks
        create_default_hooks(self.hooks)

        # Load hook config if exists
        hook_config = Path.home() / ".aria" / "hooks.yaml"
        if hook_config.exists():
            self.hooks.load_from_config(hook_config)

        # Load all skills (built-in and user)
        try:
            self.skill_loader.load_all()
        except Exception as e:
            print(f"Warning: Error loading skills: {e}")

    def _register_tool_handlers(self):
        """Register handlers for all tools that Claude can use via Tool Search."""
        tsm = self.tool_search_manager

        # Core control tools
        tsm.register_tool_handler("click", self._tool_click)
        tsm.register_tool_handler("double_click", self._tool_double_click)
        tsm.register_tool_handler("type_text", self._tool_type_text)
        tsm.register_tool_handler("press_key", self._tool_press_key)
        tsm.register_tool_handler("hotkey", self._tool_hotkey)
        tsm.register_tool_handler("scroll", self._tool_scroll)
        tsm.register_tool_handler("open_app", self._tool_open_app)
        tsm.register_tool_handler("open_url", self._tool_open_url)

        # Screen tools
        tsm.register_tool_handler("capture_screen", self._tool_capture_screen)
        tsm.register_tool_handler("get_active_app", self._tool_get_active_app)

        # Memory tools
        tsm.register_tool_handler("remember", self._tool_remember)
        tsm.register_tool_handler("recall", self._tool_recall)
        tsm.register_tool_handler("list_all_memories", self._tool_list_memories)

        # Voice tools
        tsm.register_tool_handler("speak", self._tool_speak)

    # =========================================================================
    # Tool Handlers (for Tool Search integration)
    # =========================================================================

    def _tool_click(self, x: int, y: int, button: str = "left") -> Dict[str, Any]:
        """Handle click tool."""
        scaled_x, scaled_y = self._scale_coordinates(x, y)
        if button == "right":
            success = self.control.right_click(scaled_x, scaled_y)
        else:
            success = self.control.click(scaled_x, scaled_y)
        return {"success": success, "clicked_at": [scaled_x, scaled_y]}

    def _tool_double_click(self, x: int, y: int) -> Dict[str, Any]:
        """Handle double click tool."""
        scaled_x, scaled_y = self._scale_coordinates(x, y)
        success = self.control.double_click(scaled_x, scaled_y)
        return {"success": success, "clicked_at": [scaled_x, scaled_y]}

    def _tool_type_text(self, text: str) -> Dict[str, Any]:
        """Handle type text tool."""
        success = self.control.type_text(text)
        return {"success": success, "typed": text[:50] + "..." if len(text) > 50 else text}

    def _tool_press_key(self, key: str) -> Dict[str, Any]:
        """Handle press key tool."""
        success = self.control.press_key(key)
        return {"success": success, "key": key}

    def _tool_hotkey(self, keys: List[str]) -> Dict[str, Any]:
        """Handle hotkey tool."""
        success = self.control.hotkey(*keys)
        return {"success": success, "keys": keys}

    def _tool_scroll(self, amount: int, x: int = None, y: int = None) -> Dict[str, Any]:
        """Handle scroll tool."""
        success = self.control.scroll(amount, x, y)
        return {"success": success, "scrolled": amount}

    def _tool_open_app(self, app: str) -> Dict[str, Any]:
        """Handle open app tool."""
        success = self.control.open_app(app)
        return {"success": success, "app": app}

    def _tool_open_url(self, url: str) -> Dict[str, Any]:
        """Handle open URL tool."""
        success = self.control.open_url(url)
        return {"success": success, "url": url}

    def _tool_capture_screen(self) -> Dict[str, Any]:
        """Handle screen capture tool."""
        screenshot_b64, image_size = self.screen.capture_to_base64_with_size()
        if screenshot_b64:
            self.image_size_for_claude = image_size
            return {
                "success": True,
                "image_base64": screenshot_b64,
                "size": list(image_size)
            }
        return {"success": False, "error": "Failed to capture screen"}

    def _tool_get_active_app(self) -> Dict[str, Any]:
        """Handle get active app tool."""
        if hasattr(self.control, 'get_active_app'):
            app_name = self.control.get_active_app()
            return {"success": True, "active_app": app_name}
        return {"success": False, "error": "get_active_app not available"}

    def _tool_remember(self, fact: str, category: str = "other") -> Dict[str, Any]:
        """Handle remember tool."""
        try:
            self.memory.add_fact(fact, category)
            return {"success": True, "stored": fact, "category": category}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _tool_recall(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Handle recall tool."""
        try:
            results = self.memory.search_memories(query, n_results=n_results)
            return {"success": True, "memories": results}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _tool_list_memories(self) -> Dict[str, Any]:
        """Handle list all memories tool."""
        try:
            facts = self.memory.get_all_facts()
            return {"success": True, "facts": facts, "count": len(facts)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _tool_speak(self, text: str) -> Dict[str, Any]:
        """Handle speak tool (TTS)."""
        # Voice TTS will be handled by the main app
        return {"success": True, "speak": text}

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by name with the given input.

        Returns:
            Result dictionary from the tool handler
        """
        handler = self.tool_search_manager.get_tool_handler(tool_name)
        if handler:
            try:
                return handler(**tool_input)
            except Exception as e:
                return {"success": False, "error": f"Tool execution error: {e}"}
        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

    def _call_claude_with_tools(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        include_screen: bool = True,
        use_tool_search: bool = True
    ) -> Dict[str, Any]:
        """
        Call Claude with native tool support via Tool Search.

        Args:
            messages: Conversation messages
            system_prompt: System prompt to use
            include_screen: Whether to capture and include screen
            use_tool_search: Whether to use tool search (defer non-essential tools)

        Returns:
            Dict with 'response', 'tool_calls', and 'stop_reason'
        """
        # Get tools from manager
        tools = self.tool_search_manager.get_all_tools_for_api(include_search=use_tool_search)

        # Add screen capture to first message if needed
        if include_screen and messages:
            screenshot_b64, image_size = self.screen.capture_to_base64_with_size()
            if screenshot_b64:
                self.image_size_for_claude = image_size
                # Prepend image to first user message
                first_msg = messages[0]
                if first_msg["role"] == "user":
                    if isinstance(first_msg["content"], str):
                        first_msg["content"] = [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": screenshot_b64
                                }
                            },
                            {"type": "text", "text": first_msg["content"]}
                        ]

        try:
            # Call Claude with tools and beta header
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=CLAUDE_MAX_TOKENS,
                system=system_prompt,
                messages=messages,
                tools=tools,
                betas=[TOOL_SEARCH_BETA]  # Enable tool search beta
            )

            # Parse response
            text_content = ""
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    text_content += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input
                    })

            return {
                "response": text_content,
                "tool_calls": tool_calls,
                "stop_reason": response.stop_reason
            }

        except Exception as e:
            print(f"Claude API error: {e}")
            return {
                "response": f"Sorry, I encountered an error: {e}",
                "tool_calls": [],
                "stop_reason": "error"
            }

    def process_request_with_tools(
        self,
        user_input: str,
        include_screen: bool = True,
        max_iterations: int = 10
    ) -> str:
        """
        Process a request using Claude's native tool calling with Tool Search.

        This is the new preferred method that uses Claude's tools API
        instead of parsing JSON from text responses.

        Args:
            user_input: The user's request
            include_screen: Whether to capture screen
            max_iterations: Maximum tool use iterations

        Returns:
            Final response text
        """
        self.current_task_id = hashlib.md5(f"{user_input}{time.time()}".encode()).hexdigest()[:8]

        # Build initial message
        memory_context = self.memory.get_context_for_request(user_input)
        initial_text = user_input
        if memory_context:
            initial_text = f"[MEMORY CONTEXT]\n{memory_context}\n[END MEMORY]\n\n{user_input}"

        messages = [{"role": "user", "content": initial_text}]

        # Agentic loop with tool calling
        iteration = 0
        final_response = ""

        while iteration < max_iterations:
            iteration += 1
            print(f"[{self.current_task_id}:{iteration}] Calling Claude with tools...")

            # Call Claude
            result = self._call_claude_with_tools(
                messages=messages,
                system_prompt=ARIA_SYSTEM_PROMPT,
                include_screen=(include_screen and iteration == 1),
                use_tool_search=True
            )

            final_response = result["response"]
            tool_calls = result["tool_calls"]
            stop_reason = result["stop_reason"]

            # If no tool use, we're done
            if stop_reason != "tool_use" or not tool_calls:
                print(f"[{self.current_task_id}] Completed: {stop_reason}")
                break

            # Execute tools and build tool results
            tool_results = []
            for tool_call in tool_calls:
                print(f"[{self.current_task_id}] Executing tool: {tool_call['name']}")
                tool_result = self._execute_tool(tool_call["name"], tool_call["input"])

                # Format result for Claude
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call["id"],
                    "content": json.dumps(tool_result)
                })

                # Small delay for UI actions
                if tool_call["name"] in ["click", "double_click", "type_text", "scroll"]:
                    time.sleep(ACTION_DELAY_MS / 1000.0)

            # Add assistant message with tool use
            assistant_content = []
            if final_response:
                assistant_content.append({"type": "text", "text": final_response})
            for tc in tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"]
                })
            messages.append({
                "role": "assistant",
                "content": assistant_content
            })

            # Add tool results
            messages.append({
                "role": "user",
                "content": tool_results
            })

        # Store interaction
        import threading
        threading.Thread(
            target=self.memory.extract_and_store_memories,
            args=(user_input, final_response, []),
            daemon=True
        ).start()

        return final_response

    # =========================================================================
    # MCP (Model Context Protocol) Client Methods
    # =========================================================================

    def get_mcp_tools(self, server_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of available tools from MCP servers.

        Args:
            server_name: Optional specific server, or None for all servers

        Returns:
            List of tool definitions with server name included
        """
        return self.mcp_client.list_tools(server_name)

    def call_mcp_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        """
        Call a tool on an MCP server.

        Args:
            server_name: Name of the server (e.g., 'filesystem', 'fetch')
            tool_name: Name of the tool to invoke
            arguments: Arguments to pass to the tool

        Returns:
            Tool result dictionary with 'content' or 'error'
        """
        return self.mcp_client.call_tool(server_name, tool_name, arguments)

    def start_mcp_server(self, server_name: str) -> bool:
        """
        Start a specific MCP server.

        Args:
            server_name: Name of the server to start

        Returns:
            True if server started successfully
        """
        return self.mcp_client.start_server(server_name)

    def stop_mcp_server(self, server_name: str):
        """
        Stop a specific MCP server.

        Args:
            server_name: Name of the server to stop
        """
        self.mcp_client.stop_server(server_name)

    def get_mcp_server_status(self, server_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of MCP servers.

        Args:
            server_name: Specific server name, or None for all servers

        Returns:
            Status dictionary for the server(s)
        """
        if server_name:
            return self.mcp_client.get_server_status(server_name)
        return self.mcp_client.get_all_server_status()

    async def _try_skill_match(self, user_input: str) -> Optional[str]:
        """
        Try to match user input to a skill and execute it.

        Returns response if skill matched and executed, None otherwise.
        """
        # Find matching skills
        matches = self.skill_registry.find_matching(user_input, min_score=0.5)

        if not matches:
            return None

        skill, score = matches[0]
        print(f"[{self.current_task_id}] Skill match: {skill.name} (score: {score:.2f})")

        # Build context
        context = SkillContext(
            user_input=user_input,
            memory_context=self.memory.get_context_for_request(user_input),
            active_app=self.control.get_active_app() if hasattr(self.control, 'get_active_app') else None,
        )

        # Capture screen if needed
        if skill.requires_screen:
            screenshot, _ = self.screen.capture_to_base64_with_size()
            context.screen_base64 = screenshot

        # Execute skill
        try:
            result = await skill.execute(context)
        except Exception as e:
            print(f"Skill execution error: {e}")
            return None

        if not result.success:
            print(f"Skill failed: {result.error}")
            return None

        # Handle handoffs
        if result.handoff_to:
            next_skill = self.skill_registry.get(result.handoff_to)
            if next_skill:
                print(f"Handing off to: {result.handoff_to}")
                return await self._try_skill_match(user_input)  # Retry with new context

        # Handle confirmation requests
        if result.needs_confirmation:
            return result.confirmation_prompt

        # Handle markdown skill instructions
        if result.data and "instructions" in result.data:
            # Return instructions for Claude to follow
            return None  # Let normal flow handle with instructions in context

        return result.output

    def _run_skill_match(self, user_input: str) -> Optional[str]:
        """
        Synchronous wrapper to run async skill matching.
        """
        try:
            # Try to get existing loop
            try:
                loop = asyncio.get_running_loop()
                # Already in async context, create task
                future = asyncio.ensure_future(self._try_skill_match(user_input))
                return loop.run_until_complete(future)
            except RuntimeError:
                # No running loop, create new one
                return asyncio.run(self._try_skill_match(user_input))
        except Exception as e:
            print(f"Skill match error: {e}")
            return None

    def process_request(self, user_input: str, include_screen: bool = True, max_iterations: int = 5) -> str:
        """
        Process a user request with intelligent classification.

        NEW FLOW:
        1. Classify request type (question/conversation vs action)
        2. For questions/conversation: Use multi-model reasoning
        3. For actions: Use action execution flow
        """
        loop_start = time.time()

        # Generate task ID for tracking
        self.current_task_id = hashlib.md5(f"{user_input}{time.time()}".encode()).hexdigest()[:8]

        # Check if this is a coding request - delegate to Claude Code
        if self.claude_bridge.is_coding_request(user_input):
            print(f"[{self.current_task_id}] Detected coding request, delegating to Claude Code...")
            return self._handle_coding_request(user_input)

        # Step 1: INTELLIGENT CLASSIFICATION
        # This is the key change - determine if user wants conversation or action
        memory_context = self.memory.get_context_for_request(user_input)
        classification = self.reasoner.classify_request(user_input, memory_context)

        print(f"[{self.current_task_id}] Request type: {classification.type.value} "
              f"(confidence: {classification.confidence:.2f}, action: {classification.requires_action})")

        # Step 2: ROUTE BASED ON CLASSIFICATION

        # Path A: CONVERSATION (questions, chat, explanations, opinions)
        if classification.type in [RequestType.QUESTION, RequestType.CONVERSATION,
                                    RequestType.EXPLANATION, RequestType.OPINION]:
            return self._handle_conversation(user_input, memory_context, include_screen)

        # Path B: CONFIRMATION - short response
        if classification.type == RequestType.CONFIRMATION:
            return "Got it!"

        # Path C: CODING - delegate to Claude Code
        if classification.type == RequestType.CODING:
            print(f"[{self.current_task_id}] Routing to Claude Code...")
            return self._handle_coding_request(user_input)

        # Path D: ACTION - use existing action flow
        if classification.type == RequestType.ACTION or classification.requires_action:
            return self._handle_action_request(user_input, include_screen, max_iterations, loop_start)

        # Path E: UNKNOWN - try conversation first (safer default)
        print(f"[{self.current_task_id}] Unknown type, defaulting to conversation")
        return self._handle_conversation(user_input, memory_context, include_screen)

    def _handle_conversation(self, user_input: str, memory_context: str, include_screen: bool = False) -> str:
        """
        Handle conversational requests with intelligent multi-model reasoning.

        This is where Aria's intelligence shines - natural, helpful responses.
        """
        print(f"[{self.current_task_id}] Generating intelligent response...")

        # Get screen context if useful for the conversation
        screen_context = ""
        if include_screen and any(word in user_input.lower() for word in
                                   ["screen", "see", "looking at", "what's", "this", "that"]):
            screen_context = self.vision.get_screen_context()
            memory_context = f"{memory_context}\n\nCurrent screen: {screen_context}"

        # Use multi-model reasoning for intelligent response
        result = self.reasoner.reason_sync(
            user_input=user_input,
            memory_context=memory_context,
            use_multi_model=True
        )

        print(f"[{self.current_task_id}] Response generated using: {', '.join(result.models_used)}")

        # Store interaction in memory for future context
        import threading
        threading.Thread(
            target=self.memory.extract_and_store_memories,
            args=(user_input, result.response, []),
            daemon=True
        ).start()

        return result.response

    def _handle_action_request(self, user_input: str, include_screen: bool,
                                max_iterations: int, loop_start: float) -> str:
        """
        Handle action requests using the execution flow.

        This is the original action-oriented flow, used only when user
        explicitly wants to perform an action on their computer.
        """
        # Quick classify for simple commands
        quick = self.intent_engine.quick_classify(user_input)
        if quick["type"] == "simple_action" and quick["confidence"] > 0.9:
            print(f"[{self.current_task_id}] Quick action: {quick['action']}")
            self._execute_action(quick["action"])
            self._record_success(user_input, int((time.time() - loop_start) * 1000), 1, 1)
            return "Done."

        if quick["type"] == "greeting":
            return "Hey! What can I help you with?"

        if quick["type"] == "exit":
            return "Goodbye!"

        # Try skill matching for registered skills
        skill_result = self._run_skill_match(user_input)
        if skill_result:
            self._record_success(user_input, int((time.time() - loop_start) * 1000), 1, 1)
            return skill_result

        # Understand intent with memory context
        print(f"[{self.current_task_id}] Understanding action intent...")
        memory_facts = self.memory.search_memories(user_input, n_results=5)
        intent = self.intent_engine.understand(user_input, memory_facts)

        print(f"[{self.current_task_id}] Intent: {intent.goal} (confidence: {intent.confidence:.2f})")
        if intent.resolved_references:
            print(f"[{self.current_task_id}] Resolved: {intent.resolved_references}")

        # Check if clarification needed (before planning)
        clarification = self.clarification.should_ask(intent, failure_count=self.failure_count)
        if clarification:
            print(f"[{self.current_task_id}] Asking clarification: {clarification.reason}")
            self.failure_count = 0  # Reset on clarification
            return clarification.question

        # Plan the task
        print(f"[{self.current_task_id}] Planning action...")

        # Check for successful past approaches
        best_approach = self.learning.get_best_approach(intent.goal)
        if best_approach and best_approach["success_rate"] > 0.7:
            print(f"[{self.current_task_id}] Using proven approach (success rate: {best_approach['success_rate']:.0%})")

        # Get screen description if needed
        screen_description = None
        if intent.requires_screen:
            screen_description = self.vision.get_screen_context()

        plan = self.planner.plan(intent, screen_description)
        print(f"[{self.current_task_id}] Plan: {plan.complexity} ({len(plan.steps)} steps)")

        # Execute with verification
        result = self._execute_plan(user_input, intent, plan, include_screen, max_iterations)

        # Learn from outcome
        total_time = int((time.time() - loop_start) * 1000)
        self._record_outcome(intent.goal, result, total_time, plan)

        return result["response"]

    def _execute_plan(
        self,
        user_input: str,
        intent: Intent,
        plan: Plan,
        include_screen: bool,
        max_iterations: int
    ) -> Dict[str, Any]:
        """Execute a plan with the agentic loop and verification."""
        all_responses = []
        iteration = 0
        task_done = False
        steps_attempted = 0
        steps_succeeded = 0

        # Get memory context once
        memory_context = self.memory.get_context_for_request(user_input)

        while iteration < max_iterations and not task_done:
            iteration += 1
            iter_start = time.time()

            # Build the message content
            content = []
            screenshot_b64 = None

            # Always capture screenshot for iterations 2+ (need to verify actions)
            should_capture = include_screen or iteration > 1

            if should_capture:
                screenshot_b64, image_size = self.screen.capture_to_base64_with_size()
                if screenshot_b64:
                    self.image_size_for_claude = image_size
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": screenshot_b64
                        }
                    })

            # Build the message text with plan context
            screen_info = f"\n[Screen: {self.image_size_for_claude[0]}x{self.image_size_for_claude[1]}]" if screenshot_b64 else ""

            if iteration == 1:
                # First iteration: include full context
                plan_info = ""
                if plan.steps:
                    plan_info = f"\n[PLAN: {plan.complexity} task - {len(plan.steps)} steps]"

                full_message = f"{intent.goal}{screen_info}{plan_info}"
                if memory_context:
                    full_message = f"[MEMORY CONTEXT]\n{memory_context}\n[END MEMORY]\n\n{full_message}"
            else:
                # Follow-up iterations: just update on progress
                full_message = f"[Screen updated]{screen_info}\nContinue: {intent.goal}"

            content.append({"type": "text", "text": full_message})
            self.conversation_history.append({"role": "user", "content": content})

            # Trim history to prevent context overflow
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            try:
                # Use Haiku for speed on follow-up iterations
                model = CLAUDE_MODEL if iteration == 1 else CLAUDE_MODEL_FAST

                # Call Claude with ACTION-focused prompt (since we're executing an action)
                response = self.client.messages.create(
                    model=model,
                    max_tokens=300,
                    system=ARIA_ACTION_PROMPT,
                    messages=self.conversation_history
                )
                assistant_message = response.content[0].text
                self.conversation_history.append({"role": "assistant", "content": assistant_message})

                # Parse response
                text_response, actions = self._parse_response(assistant_message)
                if text_response:
                    all_responses.append(text_response)

                # Check if done
                for action in actions:
                    if action.get("done"):
                        task_done = True
                        steps_succeeded += 1
                        break

                # Execute actions
                executable_actions = [a for a in actions if a.get("action")]
                if executable_actions:
                    for action in executable_actions:
                        steps_attempted += 1
                        success = self._execute_action(action)
                        if success:
                            steps_succeeded += 1

                    # Short delay for UI to settle
                    if not task_done:
                        time.sleep(ACTION_DELAY_MS / 1000.0)
                elif not task_done and not actions:
                    # No actions and not done - might be stuck
                    self.failure_count += 1
                    break

                iter_time = (time.time() - iter_start) * 1000
                model_name = model.split('-')[1] if '-' in model else model[:6]
                print(f"[{self.current_task_id}:{iteration}] {model_name}: {iter_time:.0f}ms | {text_response[:40] if text_response else 'action'}...")

            except Exception as e:
                print(f"[{self.current_task_id}] Error: {e}")
                all_responses.append(f"Sorry, something went wrong.")
                self.failure_count += 1
                break

        print(f"[{self.current_task_id}] Completed in {iteration} iterations ({steps_succeeded}/{steps_attempted} steps)")

        # Reset failure count on success
        if task_done:
            self.failure_count = 0

        return {
            "success": task_done,
            "response": all_responses[-1] if all_responses else "Done.",
            "iterations": iteration,
            "steps_attempted": steps_attempted,
            "steps_succeeded": steps_succeeded
        }

    def _record_outcome(self, goal: str, result: Dict, duration_ms: int, plan: Plan):
        """Record task outcome for learning."""
        self.learning.record_outcome(
            goal=goal,
            success=result["success"],
            duration_ms=duration_ms,
            steps_attempted=result["steps_attempted"],
            steps_succeeded=result["steps_succeeded"],
            failure_reason=None if result["success"] else "Task not completed",
            approach_id=f"plan_{plan.complexity}",
            plan=plan
        )

        # Store memories in background
        import threading
        threading.Thread(
            target=self.memory.extract_and_store_memories,
            args=(goal, result["response"], []),
            daemon=True
        ).start()

    def _record_success(self, goal: str, duration_ms: int, attempted: int, succeeded: int):
        """Record a quick success."""
        self.learning.record_outcome(
            goal=goal,
            success=True,
            duration_ms=duration_ms,
            steps_attempted=attempted,
            steps_succeeded=succeeded,
            approach_id="quick_action"
        )

    def handle_clarification_response(self, user_response: str, original_request: str) -> str:
        """
        Handle a user's response to a clarification question.

        Args:
            user_response: What the user said in response
            original_request: The original request that triggered clarification

        Returns:
            Response to speak to user, or continues with task
        """
        # Interpret the response
        interpreted = self.clarification.interpret_response(
            Clarification(question="", options=[], reason=""),  # Dummy for interpretation
            user_response
        )

        if interpreted["action"] == "proceed":
            # User confirmed - retry with full context
            return self.process_request(original_request)

        elif interpreted["action"] == "cancel":
            self.failure_count = 0
            return "Okay, cancelled."

        elif interpreted["action"] == "retry":
            self.failure_count = 0
            return self.process_request(original_request)

        elif interpreted["action"] == "wait_for_explanation":
            return "Okay, tell me more."

        elif interpreted["action"] == "wait_for_visual":
            return "Okay, show me what you mean."

        else:
            # Interpret as additional input - combine with original
            combined = f"{original_request} - specifically: {user_response}"
            return self.process_request(combined)

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about learning performance."""
        return {
            "total_outcomes": len(self.learning.outcomes),
            "overall_success_rate": self.learning.get_success_rate(),
            "approaches": len(self.learning.approaches),
            "patterns_learned": len(self.learning.patterns),
            "suggestions": self.learning.suggest_improvements()
        }

    def get_skills(self) -> List[Dict[str, Any]]:
        """Get list of all available skills."""
        skills = []
        for skill in self.skill_registry.all():
            skills.append({
                "name": skill.name,
                "description": skill.description,
                "triggers": skill.triggers,
                "category": skill.category.value,
                "type": "python" if skill.is_python_skill() else "markdown",
                "is_user_skill": skill.is_user_skill,
            })
        return skills

    def reload_user_skills(self) -> int:
        """Reload user skills from ~/.aria/skills/."""
        return self.skill_loader.reload_user_skills()

    def _parse_response(self, response: str) -> Tuple[str, List[dict]]:
        """Parse response text and extract any action JSON blocks."""
        actions = []
        text_parts = []

        # Split by code blocks
        parts = response.split("```")

        for i, part in enumerate(parts):
            if i % 2 == 0:
                # Text part
                text_parts.append(part.strip())
            else:
                # Code block - check if JSON
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    action = json.loads(part)
                    if isinstance(action, dict) and "action" in action:
                        actions.append(action)
                except json.JSONDecodeError:
                    # Not valid JSON, treat as text
                    text_parts.append(part)

        text_response = " ".join(filter(None, text_parts))
        return text_response, actions

    def _scale_coordinates(self, x: int, y: int) -> tuple:
        """Scale coordinates from Claude's image space to logical screen space."""
        # Claude sees a 1920xH image, but pyautogui uses logical screen coordinates
        scale_x = self.logical_screen_size[0] / self.image_size_for_claude[0]
        scale_y = self.logical_screen_size[1] / self.image_size_for_claude[1]

        scaled_x = int(x * scale_x)
        scaled_y = int(y * scale_y)

        print(f"Scaling coordinates: ({x}, {y}) -> ({scaled_x}, {scaled_y})")
        print(f"  Scale factors: x={scale_x:.2f}, y={scale_y:.2f}")

        return scaled_x, scaled_y

    def _execute_action(self, action: dict) -> bool:
        """Execute a single action."""
        action_type = action.get("action")

        try:
            if action_type == "click":
                x, y = self._scale_coordinates(action["x"], action["y"])
                return self.control.click(x, y)

            elif action_type == "double_click":
                x, y = self._scale_coordinates(action["x"], action["y"])
                return self.control.double_click(x, y)

            elif action_type == "right_click":
                x, y = self._scale_coordinates(action["x"], action["y"])
                return self.control.right_click(x, y)

            elif action_type == "type":
                return self.control.type_text(action["text"])

            elif action_type == "press":
                return self.control.press_key(action["key"])

            elif action_type == "hotkey":
                return self.control.hotkey(*action["keys"])

            elif action_type == "scroll":
                x = action.get("x")
                y = action.get("y")
                return self.control.scroll(action["amount"], x, y)

            elif action_type == "open_app":
                return self.control.open_app(action["app"])

            elif action_type == "open_url":
                return self.control.open_url(action["url"])

            elif action_type == "wait":
                import time
                time.sleep(action.get("seconds", 1))
                return True

            else:
                print(f"Unknown action type: {action_type}")
                return False

        except Exception as e:
            print(f"Action execution error: {e}")
            return False

    def _handle_coding_request(self, user_input: str) -> str:
        """
        Handle a coding request by delegating to Claude Code.

        Args:
            user_input: The coding-related request

        Returns:
            Voice-friendly summary of what Claude Code did
        """
        try:
            # Run Claude Code
            print(f"Running Claude Code with: {user_input}")
            output = self.claude_bridge.run_claude(user_input)
            print(f"Claude Code output: {output[:200]}...")

            # Summarize for voice
            summary = self.claude_bridge.summarize_for_voice(output)

            # Store in memory
            self.memory.remember_interaction(
                f"Coding: {user_input[:50]}",
                user_input,
                summary
            )

            return summary

        except Exception as e:
            print(f"Claude Code error: {e}")
            return f"Sorry, I couldn't complete that coding task. Error: {str(e)[:50]}"

    def set_project(self, path: str) -> bool:
        """Set the current project for Claude Code."""
        return self.claude_bridge.set_project(path)

    def get_screen_context(self) -> str:
        """Get current screen context for voice response."""
        return self.vision.get_screen_context()

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    # =========================================================================
    # Gesture Recognition Control
    # =========================================================================

    def enable_gestures(self, auto_confirm: bool = True, auto_stop: bool = True) -> bool:
        """
        Enable gesture recognition for hands-free control.

        Args:
            auto_confirm: Use thumbs up gesture for confirmation
            auto_stop: Use open palm gesture to stop/cancel operations

        Returns:
            True if gestures were enabled successfully
        """
        try:
            self.gesture_controller = get_gesture_controller()

            # Set up gesture action callbacks
            if auto_confirm:
                self.gesture_controller.set_action_callback(
                    GestureAction.CONFIRM,
                    self._on_gesture_confirm
                )

            if auto_stop:
                self.gesture_controller.set_action_callback(
                    GestureAction.STOP,
                    self._on_gesture_stop
                )
                self.gesture_controller.set_action_callback(
                    GestureAction.CANCEL,
                    self._on_gesture_cancel
                )

            # Start gesture recognition
            self.gesture_controller.start()
            self.gesture_enabled = True
            print("Gesture recognition enabled")
            return True

        except Exception as e:
            print(f"Failed to enable gestures: {e}")
            return False

    def disable_gestures(self):
        """Disable gesture recognition."""
        if self.gesture_controller:
            self.gesture_controller.stop()
            self.gesture_enabled = False
            print("Gesture recognition disabled")

    def _on_gesture_confirm(self):
        """Handle thumbs up gesture for confirmation."""
        print("Gesture: Thumbs up (confirm)")

        if self._pending_gesture_confirmation:
            # Complete pending confirmation
            callback = self._pending_gesture_confirmation.get("callback")
            if callback:
                callback(True)
            self._pending_gesture_confirmation = None

    def _on_gesture_stop(self):
        """Handle open palm gesture for stop."""
        print("Gesture: Open palm (stop)")

        # Cancel any pending confirmation
        if self._pending_gesture_confirmation:
            callback = self._pending_gesture_confirmation.get("callback")
            if callback:
                callback(False)
            self._pending_gesture_confirmation = None

        # Reset current task
        self.failure_count = 0
        self.current_task_id = None

    def _on_gesture_cancel(self):
        """Handle thumbs down gesture for cancel."""
        print("Gesture: Thumbs down (cancel)")

        # Cancel any pending confirmation
        if self._pending_gesture_confirmation:
            callback = self._pending_gesture_confirmation.get("callback")
            if callback:
                callback(False)
            self._pending_gesture_confirmation = None

    def request_gesture_confirmation(
        self,
        prompt: str,
        callback: Optional[Any] = None,
        timeout: float = 10.0
    ) -> bool:
        """
        Request confirmation via gesture (thumbs up/down).

        Args:
            prompt: Message to display/speak to user
            callback: Optional callback function(confirmed: bool)
            timeout: Seconds to wait for gesture

        Returns:
            True if confirmation was requested, False if gestures not enabled
        """
        if not self.gesture_enabled or not self.gesture_controller:
            return False

        self._pending_gesture_confirmation = {
            "prompt": prompt,
            "callback": callback,
            "timeout": timeout
        }

        # Use the gesture controller's built-in confirmation
        if callback:
            self.gesture_controller.request_confirmation(callback, timeout)

        return True

    def is_gesture_enabled(self) -> bool:
        """Check if gesture recognition is currently enabled."""
        return self.gesture_enabled and self.gesture_controller is not None


# Singleton
_agent: Optional[AriaAgent] = None


def get_agent() -> AriaAgent:
    """Get the singleton AriaAgent instance."""
    global _agent
    if _agent is None:
        _agent = AriaAgent()
    return _agent
