"""
Tool Search Integration for Aria

Implements Anthropic's MCP Tool Search feature to efficiently manage
hundreds of tools by dynamically loading them on-demand.

Key concepts:
- Core tools: Always loaded (click, type, scroll, etc.)
- Deferred tools: Loaded via search when needed (MCP tools, specialized skills)
- Tool Search uses BM25 (natural language) for voice-friendly queries
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class ToolCategory(Enum):
    """Categories for tool organization."""
    CORE = "core"           # Always loaded - essential controls
    SCREEN = "screen"       # Screen capture and vision
    MEMORY = "memory"       # Memory operations
    VOICE = "voice"         # Voice/TTS
    MCP = "mcp"             # External MCP server tools
    SKILL = "skill"         # User-defined skills
    WORKFLOW = "workflow"   # Multi-step workflows


@dataclass
class ToolDefinition:
    """Complete tool definition for Claude API."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    category: ToolCategory = ToolCategory.CORE
    defer_loading: bool = False

    def to_api_schema(self) -> Dict[str, Any]:
        """Convert to Claude API tool format."""
        schema = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
        if self.defer_loading:
            schema["defer_loading"] = True
        return schema


# =============================================================================
# CORE TOOLS - Always loaded, essential for basic operation
# =============================================================================

CORE_TOOLS: List[ToolDefinition] = [
    ToolDefinition(
        name="click",
        description="Click at specific screen coordinates. Use for clicking buttons, links, icons.",
        input_schema={
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coordinate on screen"},
                "y": {"type": "integer", "description": "Y coordinate on screen"},
                "button": {
                    "type": "string",
                    "enum": ["left", "right", "middle"],
                    "default": "left",
                    "description": "Mouse button to click"
                }
            },
            "required": ["x", "y"]
        },
        category=ToolCategory.CORE,
        defer_loading=False
    ),
    ToolDefinition(
        name="double_click",
        description="Double-click at screen coordinates. Use for opening files, selecting words.",
        input_schema={
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coordinate"},
                "y": {"type": "integer", "description": "Y coordinate"}
            },
            "required": ["x", "y"]
        },
        category=ToolCategory.CORE,
        defer_loading=False
    ),
    ToolDefinition(
        name="type_text",
        description="Type text at the current cursor position. Uses clipboard for reliability.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to type"}
            },
            "required": ["text"]
        },
        category=ToolCategory.CORE,
        defer_loading=False
    ),
    ToolDefinition(
        name="press_key",
        description="Press a keyboard key. Keys: enter, tab, escape, space, backspace, delete, up, down, left, right, home, end, pageup, pagedown.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key to press"}
            },
            "required": ["key"]
        },
        category=ToolCategory.CORE,
        defer_loading=False
    ),
    ToolDefinition(
        name="hotkey",
        description="Press a keyboard shortcut. Example: ['command', 'c'] for copy, ['command', 'v'] for paste.",
        input_schema={
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keys to press together"
                }
            },
            "required": ["keys"]
        },
        category=ToolCategory.CORE,
        defer_loading=False
    ),
    ToolDefinition(
        name="scroll",
        description="Scroll the mouse wheel. Positive = up, negative = down.",
        input_schema={
            "type": "object",
            "properties": {
                "amount": {
                    "type": "integer",
                    "description": "Scroll amount (positive=up, negative=down)"
                },
                "x": {"type": "integer", "description": "Optional X coordinate to scroll at"},
                "y": {"type": "integer", "description": "Optional Y coordinate to scroll at"}
            },
            "required": ["amount"]
        },
        category=ToolCategory.CORE,
        defer_loading=False
    ),
    ToolDefinition(
        name="open_app",
        description="Open an application by name. Examples: 'Safari', 'Terminal', 'VS Code', 'Finder'.",
        input_schema={
            "type": "object",
            "properties": {
                "app": {"type": "string", "description": "Application name to open"}
            },
            "required": ["app"]
        },
        category=ToolCategory.CORE,
        defer_loading=False
    ),
    ToolDefinition(
        name="open_url",
        description="Open a URL in the default web browser.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to open"}
            },
            "required": ["url"]
        },
        category=ToolCategory.CORE,
        defer_loading=False
    ),
]


# =============================================================================
# DEFERRED TOOLS - Loaded on-demand via tool search
# =============================================================================

SCREEN_TOOLS: List[ToolDefinition] = [
    ToolDefinition(
        name="capture_screen",
        description="Capture the current screen and return as base64 PNG image.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": []
        },
        category=ToolCategory.SCREEN,
        defer_loading=True
    ),
    ToolDefinition(
        name="get_active_app",
        description="Get the name of the currently focused/active application.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": []
        },
        category=ToolCategory.SCREEN,
        defer_loading=True
    ),
]

MEMORY_TOOLS: List[ToolDefinition] = [
    ToolDefinition(
        name="remember",
        description="Store a fact in long-term memory. Categories: preference, personal, work, habit, project, other.",
        input_schema={
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "The fact to remember"},
                "category": {
                    "type": "string",
                    "enum": ["preference", "personal", "work", "habit", "project", "other"],
                    "default": "other",
                    "description": "Category for the fact"
                }
            },
            "required": ["fact"]
        },
        category=ToolCategory.MEMORY,
        defer_loading=True
    ),
    ToolDefinition(
        name="recall",
        description="Search long-term memory for relevant information.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "n_results": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of results to return"
                }
            },
            "required": ["query"]
        },
        category=ToolCategory.MEMORY,
        defer_loading=True
    ),
    ToolDefinition(
        name="list_all_memories",
        description="List all stored facts in memory.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": []
        },
        category=ToolCategory.MEMORY,
        defer_loading=True
    ),
]

VOICE_TOOLS: List[ToolDefinition] = [
    ToolDefinition(
        name="speak",
        description="Speak text aloud using text-to-speech.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to speak"}
            },
            "required": ["text"]
        },
        category=ToolCategory.VOICE,
        defer_loading=True
    ),
]


# =============================================================================
# TOOL SEARCH CONFIGURATION
# =============================================================================

# Tool search tool definition (BM25 for natural language - better for voice)
TOOL_SEARCH_TOOL = {
    "type": "tool_search_tool_bm25_20251119",
    "name": "tool_search"
}

# Beta header required for tool search
TOOL_SEARCH_BETA = "advanced-tool-use-2025-11-20"


class ToolSearchManager:
    """
    Manages tool definitions and tool search for Aria.

    Provides:
    - Core tools (always loaded)
    - Deferred tools (loaded via search)
    - MCP tool integration
    - Tool execution routing
    """

    def __init__(self):
        self._core_tools: List[ToolDefinition] = list(CORE_TOOLS)
        self._deferred_tools: List[ToolDefinition] = []
        self._mcp_tools: List[ToolDefinition] = []

        # Add built-in deferred tools
        self._deferred_tools.extend(SCREEN_TOOLS)
        self._deferred_tools.extend(MEMORY_TOOLS)
        self._deferred_tools.extend(VOICE_TOOLS)

        self._tool_handlers: Dict[str, callable] = {}

    def register_tool_handler(self, tool_name: str, handler: callable):
        """Register a handler function for a tool."""
        self._tool_handlers[tool_name] = handler

    def register_mcp_tools(self, tools: List[Dict[str, Any]], server_name: str):
        """
        Register tools from an MCP server as deferred tools.

        Args:
            tools: List of tool definitions from MCP server
            server_name: Name of the MCP server (for namespacing)
        """
        for tool in tools:
            tool_def = ToolDefinition(
                name=f"{server_name}_{tool['name']}",
                description=f"[{server_name}] {tool.get('description', '')}",
                input_schema=tool.get('inputSchema', {"type": "object", "properties": {}}),
                category=ToolCategory.MCP,
                defer_loading=True
            )
            self._mcp_tools.append(tool_def)

    def get_all_tools_for_api(self, include_search: bool = True) -> List[Dict[str, Any]]:
        """
        Get all tool definitions formatted for Claude API.

        Returns tools with proper defer_loading flags:
        - Tool search tool (if enabled)
        - Core tools (no defer_loading)
        - Deferred tools (defer_loading: true)
        - MCP tools (defer_loading: true)
        """
        tools = []

        # Add tool search tool first (if enabled)
        if include_search:
            tools.append(TOOL_SEARCH_TOOL)

        # Add core tools (always loaded)
        for tool in self._core_tools:
            tools.append(tool.to_api_schema())

        # Add deferred tools
        for tool in self._deferred_tools:
            tools.append(tool.to_api_schema())

        # Add MCP tools
        for tool in self._mcp_tools:
            tools.append(tool.to_api_schema())

        return tools

    def get_core_tools_only(self) -> List[Dict[str, Any]]:
        """Get only core tools (no search, no deferred)."""
        return [tool.to_api_schema() for tool in self._core_tools]

    def get_tool_by_name(self, name: str) -> Optional[ToolDefinition]:
        """Find a tool by name."""
        all_tools = self._core_tools + self._deferred_tools + self._mcp_tools
        for tool in all_tools:
            if tool.name == name:
                return tool
        return None

    def get_tool_handler(self, name: str) -> Optional[callable]:
        """Get the handler for a tool."""
        return self._tool_handlers.get(name)

    def get_tool_categories_prompt(self) -> str:
        """
        Get a system prompt section describing available tool categories.
        This helps Claude know what tools are searchable.
        """
        categories = {
            "Core Controls": ["click", "type", "scroll", "keyboard shortcuts", "open apps/URLs"],
            "Screen & Vision": ["capture screen", "get active app"],
            "Memory": ["remember facts", "recall information", "list memories"],
            "Voice": ["text-to-speech"],
        }

        if self._mcp_tools:
            mcp_servers = set()
            for tool in self._mcp_tools:
                # Extract server name from tool name (format: servername_toolname)
                parts = tool.name.split("_", 1)
                if parts:
                    mcp_servers.add(parts[0])
            categories["External Services"] = list(mcp_servers)

        lines = ["You have access to tools in these categories:"]
        for category, items in categories.items():
            lines.append(f"- {category}: {', '.join(items)}")
        lines.append("\nUse tool_search to find specific tools when needed.")

        return "\n".join(lines)

    @property
    def total_tool_count(self) -> int:
        """Total number of tools available."""
        return len(self._core_tools) + len(self._deferred_tools) + len(self._mcp_tools)

    @property
    def deferred_tool_count(self) -> int:
        """Number of deferred (searchable) tools."""
        return len(self._deferred_tools) + len(self._mcp_tools)


# Singleton instance
_tool_search_manager: Optional[ToolSearchManager] = None


def get_tool_search_manager() -> ToolSearchManager:
    """Get the singleton ToolSearchManager instance."""
    global _tool_search_manager
    if _tool_search_manager is None:
        _tool_search_manager = ToolSearchManager()
    return _tool_search_manager
