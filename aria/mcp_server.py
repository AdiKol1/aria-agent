"""
Aria MCP Server

Exposes Aria's capabilities to Claude Code via Model Context Protocol.
This gives Claude Code the ability to:
- See the screen
- Control the computer (click, type, scroll, etc.)
- Speak to the user
- Access shared memory

Uses JSON-RPC 2.0 over stdio (compatible with Python 3.9+)
"""

import base64
import json
import sys
from typing import Any, Dict, List, Optional

# Import Aria components
from .vision import get_screen_capture
from .control import get_control
from .memory import get_memory
from .skills import get_registry, get_loader, SkillContext


class AriaMCPServer:
    """MCP Server exposing Aria capabilities."""

    def __init__(self):
        self.screen = get_screen_capture()
        self.control = get_control()
        self.memory = get_memory()
        self.voice = None

        # Skills system
        self.skill_registry = get_registry()
        self.skill_loader = get_loader()
        self.skill_loader.load_all()

        # Try to load voice (may fail in non-audio contexts)
        try:
            from .voice import get_voice
            self.voice = get_voice()
        except:
            pass

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return list of available tools."""
        return [
            # Screen capture
            {
                "name": "capture_screen",
                "description": "Capture the current screen and return as base64 PNG. Use this to see what's on the user's screen.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_active_app",
                "description": "Get the name of the currently focused application.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },

            # Mouse control
            {
                "name": "click",
                "description": "Click at specific screen coordinates. Use after capture_screen to click on UI elements.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "X coordinate"},
                        "y": {"type": "integer", "description": "Y coordinate"},
                        "button": {"type": "string", "enum": ["left", "right", "middle"], "default": "left"}
                    },
                    "required": ["x", "y"]
                }
            },
            {
                "name": "double_click",
                "description": "Double-click at specific screen coordinates.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "X coordinate"},
                        "y": {"type": "integer", "description": "Y coordinate"}
                    },
                    "required": ["x", "y"]
                }
            },
            {
                "name": "scroll",
                "description": "Scroll the mouse wheel. Positive = up, negative = down.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "amount": {"type": "integer", "description": "Scroll amount (positive=up, negative=down)"},
                        "x": {"type": "integer", "description": "Optional X coordinate to scroll at"},
                        "y": {"type": "integer", "description": "Optional Y coordinate to scroll at"}
                    },
                    "required": ["amount"]
                }
            },

            # Keyboard control
            {
                "name": "type_text",
                "description": "Type text at the current cursor position. Uses clipboard paste for reliability.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to type"}
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "press_key",
                "description": "Press a single key (enter, tab, escape, space, backspace, delete, up, down, left, right, etc.)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Key to press"}
                    },
                    "required": ["key"]
                }
            },
            {
                "name": "hotkey",
                "description": "Press a keyboard shortcut. Example: ['command', 'c'] for Cmd+C",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keys to press together"
                        }
                    },
                    "required": ["keys"]
                }
            },

            # App control
            {
                "name": "open_app",
                "description": "Open an application by name (e.g., 'Safari', 'Terminal', 'VS Code')",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "description": "Application name"}
                    },
                    "required": ["app"]
                }
            },
            {
                "name": "open_url",
                "description": "Open a URL in the default browser.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to open"}
                    },
                    "required": ["url"]
                }
            },

            # Memory
            {
                "name": "remember",
                "description": "Store information in long-term memory shared with Aria.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "fact": {"type": "string", "description": "The fact to remember"},
                        "category": {
                            "type": "string",
                            "enum": ["preference", "personal", "work", "habit", "project", "other"],
                            "default": "other"
                        }
                    },
                    "required": ["fact"]
                }
            },
            {
                "name": "recall",
                "description": "Search long-term memory for relevant information.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for"},
                        "n_results": {"type": "integer", "default": 5}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "list_all_memories",
                "description": "List all stored facts in memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },

            # Voice
            {
                "name": "speak",
                "description": "Speak text aloud to the user using text-to-speech.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to speak"}
                    },
                    "required": ["text"]
                }
            },

            # Skills
            {
                "name": "list_skills",
                "description": "List all available Aria skills with their descriptions and triggers.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "run_skill",
                "description": "Execute an Aria skill by name with the given input.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "skill_name": {"type": "string", "description": "Name of the skill to run"},
                        "input": {"type": "string", "description": "Input/context for the skill"}
                    },
                    "required": ["skill_name", "input"]
                }
            },
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return the result."""
        try:
            # Screen capture
            if name == "capture_screen":
                result = self.screen.capture_to_base64_with_size()
                if result:
                    b64_data, (width, height) = result
                    return {
                        "content": [
                            {"type": "text", "text": f"Screenshot captured: {width}x{height} pixels"},
                            {"type": "image", "data": b64_data, "mimeType": "image/png"}
                        ]
                    }
                return {"content": [{"type": "text", "text": "Failed to capture screen."}]}

            elif name == "get_active_app":
                app_name = self.screen.get_active_app()
                return {"content": [{"type": "text", "text": f"Active application: {app_name}"}]}

            # Mouse control
            elif name == "click":
                success = self.control.click(
                    arguments["x"],
                    arguments["y"],
                    button=arguments.get("button", "left")
                )
                return {"content": [{"type": "text", "text": f"Click: {'success' if success else 'failed'}"}]}

            elif name == "double_click":
                success = self.control.double_click(arguments["x"], arguments["y"])
                return {"content": [{"type": "text", "text": f"Double-click: {'success' if success else 'failed'}"}]}

            elif name == "scroll":
                success = self.control.scroll(
                    arguments["amount"],
                    arguments.get("x"),
                    arguments.get("y")
                )
                return {"content": [{"type": "text", "text": f"Scroll: {'success' if success else 'failed'}"}]}

            # Keyboard control
            elif name == "type_text":
                success = self.control.type_text(arguments["text"])
                return {"content": [{"type": "text", "text": f"Type: {'success' if success else 'failed'}"}]}

            elif name == "press_key":
                success = self.control.press_key(arguments["key"])
                return {"content": [{"type": "text", "text": f"Press {arguments['key']}: {'success' if success else 'failed'}"}]}

            elif name == "hotkey":
                success = self.control.hotkey(*arguments["keys"])
                return {"content": [{"type": "text", "text": f"Hotkey {'+'.join(arguments['keys'])}: {'success' if success else 'failed'}"}]}

            # App control
            elif name == "open_app":
                success = self.control.open_app(arguments["app"])
                return {"content": [{"type": "text", "text": f"Open {arguments['app']}: {'success' if success else 'failed'}"}]}

            elif name == "open_url":
                success = self.control.open_url(arguments["url"])
                return {"content": [{"type": "text", "text": f"Open URL: {'success' if success else 'failed'}"}]}

            # Memory
            elif name == "remember":
                success = self.memory.remember_fact(
                    arguments["fact"],
                    arguments.get("category", "other")
                )
                return {"content": [{"type": "text", "text": f"Remembered: {arguments['fact'][:50]}..." if success else "Failed"}]}

            elif name == "recall":
                facts = self.memory.recall_facts(
                    arguments["query"],
                    arguments.get("n_results", 5)
                )
                if facts:
                    result = "Recalled memories:\n" + "\n".join(
                        f"- [{f['category']}] {f['fact']}" for f in facts
                    )
                else:
                    result = "No relevant memories found."
                return {"content": [{"type": "text", "text": result}]}

            elif name == "list_all_memories":
                all_facts = self.memory.get_all_facts()
                if all_facts:
                    result = f"All memories ({len(all_facts)}):\n" + "\n".join(f"- {f}" for f in all_facts)
                else:
                    result = "No memories stored yet."
                return {"content": [{"type": "text", "text": result}]}

            # Voice
            elif name == "speak":
                if self.voice:
                    success = self.voice.speak(arguments["text"])
                    return {"content": [{"type": "text", "text": f"Spoke: {'success' if success else 'failed'}"}]}
                return {"content": [{"type": "text", "text": "Voice not available"}]}

            # Skills
            elif name == "list_skills":
                skills = self.skill_registry.all()
                if skills:
                    lines = [f"Available skills ({len(skills)}):"]
                    for skill in skills:
                        triggers = ", ".join(skill.triggers[:3]) if skill.triggers else "none"
                        skill_type = "python" if skill.is_python_skill() else "markdown"
                        lines.append(f"- {skill.name} [{skill.category.value}] ({skill_type})")
                        lines.append(f"  {skill.description}")
                        lines.append(f"  Triggers: {triggers}")
                    result = "\n".join(lines)
                else:
                    result = "No skills available."
                return {"content": [{"type": "text", "text": result}]}

            elif name == "run_skill":
                skill_name = arguments["skill_name"]
                user_input = arguments["input"]

                skill = self.skill_registry.get(skill_name)
                if not skill:
                    return {"content": [{"type": "text", "text": f"Skill not found: {skill_name}"}]}

                # Build context
                context = SkillContext(
                    user_input=user_input,
                    memory_context=self.memory.get_context_for_request(user_input)
                )

                # Execute skill
                import asyncio
                try:
                    result = asyncio.run(skill.execute(context))
                    if result.success:
                        return {"content": [{"type": "text", "text": result.output}]}
                    else:
                        return {"content": [{"type": "text", "text": f"Skill failed: {result.error}"}]}
                except Exception as e:
                    return {"content": [{"type": "text", "text": f"Skill execution error: {e}"}]}

            else:
                return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}]}

        except Exception as e:
            return {"content": [{"type": "text", "text": f"Error: {str(e)}"}]}

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a JSON-RPC request."""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        result = None
        error = None

        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "aria",
                        "version": "0.1.0"
                    }
                }

            elif method == "notifications/initialized":
                # No response needed for notifications
                return None

            elif method == "tools/list":
                result = {"tools": self.get_tools()}

            elif method == "tools/call":
                tool_name = params.get("name", "")
                tool_args = params.get("arguments", {})
                result = self.call_tool(tool_name, tool_args)

            else:
                error = {"code": -32601, "message": f"Method not found: {method}"}

        except Exception as e:
            error = {"code": -32603, "message": str(e)}

        # Build response
        if request_id is not None:
            response = {"jsonrpc": "2.0", "id": request_id}
            if error:
                response["error"] = error
            else:
                response["result"] = result
            return response

        return None

    def run(self):
        """Run the MCP server over stdio."""
        sys.stderr.write("Aria MCP Server starting...\n")
        sys.stderr.flush()

        while True:
            try:
                # Read a line from stdin
                line = sys.stdin.readline()
                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                # Parse JSON-RPC request
                request = json.loads(line)

                # Handle the request
                response = self.handle_request(request)

                # Send response (if any)
                if response:
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()

            except json.JSONDecodeError as e:
                sys.stderr.write(f"JSON parse error: {e}\n")
                sys.stderr.flush()
            except Exception as e:
                sys.stderr.write(f"Error: {e}\n")
                sys.stderr.flush()


def run_server():
    """Entry point for the MCP server."""
    server = AriaMCPServer()
    server.run()


if __name__ == "__main__":
    run_server()
