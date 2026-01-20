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

# LAZY IMPORTS - Heavy modules loaded on first use for fast MCP startup
# This allows Claude Code to connect quickly without 15+ second delays


class AriaMCPServer:
    """MCP Server exposing Aria capabilities.

    Uses lazy loading to ensure fast startup - heavy modules are only
    imported when their functionality is first accessed.
    """

    def __init__(self):
        # Lazy-loaded components (initialized on first access)
        self._screen = None
        self._control = None
        self._memory = None
        self._voice = None
        self._vision_context = None
        self._vision_executor = None
        self._ambient_observer = None
        self._skill_registry = None
        self._skill_loader = None
        self._ambient = None
        self._initialized = False

    def _lazy_init(self):
        """Initialize heavy components on first use."""
        if self._initialized:
            return
        self._initialized = True

        sys.stderr.write("[MCP] Initializing components...\n")
        sys.stderr.flush()

    @property
    def screen(self):
        if self._screen is None:
            from .vision import get_screen_capture
            self._screen = get_screen_capture()
        return self._screen

    @property
    def control(self):
        if self._control is None:
            from .control import get_control
            self._control = get_control()
        return self._control

    @property
    def memory(self):
        if self._memory is None:
            from .memory import get_memory
            self._memory = get_memory()
        return self._memory

    @property
    def voice(self):
        if self._voice is None:
            try:
                from .voice import get_voice
                self._voice = get_voice()
            except:
                self._voice = False  # Mark as unavailable
        return self._voice if self._voice else None

    @property
    def vision_context(self):
        if self._vision_context is None:
            from .core.vision_context import get_vision_context
            self._vision_context = get_vision_context()
        return self._vision_context

    @property
    def vision_executor(self):
        if self._vision_executor is None:
            from .core.vision_context import VisionAwareExecutor
            self._vision_executor = VisionAwareExecutor(self.vision_context, self.control)
        return self._vision_executor

    @property
    def ambient_observer(self):
        if self._ambient_observer is None:
            from .core.vision_context import get_ambient_observer
            self._ambient_observer = get_ambient_observer()
            try:
                self._ambient_observer.start()
            except Exception as e:
                sys.stderr.write(f"[MCP] Warning: Could not start ambient observer: {e}\n")
        return self._ambient_observer

    @property
    def skill_registry(self):
        if self._skill_registry is None:
            from .skills import get_registry
            self._skill_registry = get_registry()
        return self._skill_registry

    @property
    def skill_loader(self):
        if self._skill_loader is None:
            from .skills import get_loader
            self._skill_loader = get_loader()
            self._skill_loader.load_all()
        return self._skill_loader

    @property
    def ambient(self):
        if self._ambient is None:
            from .ambient import get_ambient_system
            self._ambient = get_ambient_system()
        return self._ambient

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
            {
                "name": "get_mouse_position",
                "description": "Get the current mouse cursor position. IMPORTANT: Call this BEFORE clicking to verify where the mouse is.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_screen_context",
                "description": "Get context about recent user activity - what windows they've used, recent actions, current state. Use this to understand what the user has been doing.",
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

            # Ambient Intelligence
            {
                "name": "get_briefing",
                "description": "Get a briefing of the user's ambient intelligence status - active worlds, pending insights, and things happening.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "format": {"type": "string", "enum": ["voice", "text"], "default": "text"}
                    },
                    "required": []
                }
            },
            {
                "name": "list_worlds",
                "description": "List all configured worlds (domains of work/life like 'Real Estate', 'Tech Startup', etc.).",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "create_world",
                "description": "Create a new world (domain) for ambient monitoring.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "World name (e.g., 'Real Estate')"},
                        "description": {"type": "string", "description": "What this world represents"},
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keywords that indicate this world is relevant"
                        }
                    },
                    "required": ["name", "description"]
                }
            },
            {
                "name": "add_entity",
                "description": "Add a person, company, or topic to track in a world.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "world_id": {"type": "string", "description": "ID of the world to add to"},
                        "name": {"type": "string", "description": "Entity name (e.g., 'Compass Real Estate')"},
                        "entity_type": {
                            "type": "string",
                            "enum": ["person", "company", "brand", "topic", "location", "product", "event", "hashtag", "custom"],
                            "default": "custom"
                        },
                        "relationship": {
                            "type": "string",
                            "enum": ["competitor", "client", "prospect", "partner", "investor", "influencer", "vendor", "colleague", "target", "watch", "self", "other"],
                            "default": "watch"
                        },
                        "importance": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
                        "watch_for": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Events to watch for (e.g., 'price drop', 'new listing')"
                        }
                    },
                    "required": ["world_id", "name"]
                }
            },
            {
                "name": "add_goal",
                "description": "Add a goal to track in a world.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "world_id": {"type": "string", "description": "ID of the world to add to"},
                        "description": {"type": "string", "description": "What you want to achieve"},
                        "priority": {
                            "type": "string",
                            "enum": ["critical", "high", "medium", "low"],
                            "default": "medium"
                        },
                        "deadline": {"type": "string", "description": "Optional deadline in ISO format"}
                    },
                    "required": ["world_id", "description"]
                }
            },
            {
                "name": "get_insights",
                "description": "Get pending insights that need attention.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 5},
                        "priority": {
                            "type": "string",
                            "enum": ["critical", "high", "medium", "low"],
                            "description": "Filter by priority level"
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "get_ambient_status",
                "description": "Get the status of the ambient intelligence system.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "ambient_command",
                "description": "Process a natural language command related to ambient intelligence. Use this when the user says something about their work domains, tracking competitors, setting goals, or wanting updates. Examples: 'I work in real estate', 'track Compass as a competitor', 'my goal is to close 3 deals', 'what's going on in my worlds'.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The natural language command from the user"},
                        "context": {"type": "string", "description": "Optional conversation context that might help interpret the command"}
                    },
                    "required": ["command"]
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

            elif name == "get_mouse_position":
                pos = self.vision_context.mouse_position
                return {"content": [{"type": "text", "text": f"Mouse position: x={pos[0]}, y={pos[1]}"}]}

            elif name == "get_screen_context":
                context = self.ambient_observer.get_context_summary()
                lines = [
                    "Screen Context Summary:",
                    f"  Observations: {context.get('observation_count', 0)}",
                    f"  User actions: {context.get('user_action_count', 0)}",
                    f"  Recent clicks: {context.get('recent_click_count', 0)}",
                    f"  Windows used: {', '.join(context.get('windows_used', [])) or 'None'}",
                ]
                current = context.get('current_state')
                if current:
                    lines.append(f"  Current window: {current.get('active_window', 'Unknown')}")
                    lines.append(f"  Mouse at: {current.get('mouse_position', (0,0))}")
                return {"content": [{"type": "text", "text": "\n".join(lines)}]}

            # Mouse control - NOW WITH VISION AWARENESS
            elif name == "click":
                x, y = arguments["x"], arguments["y"]
                button = arguments.get("button", "left")

                # Use vision-aware execution
                action_ctx = self.vision_executor.click(x, y, button)

                # Build detailed response
                response_parts = [f"Click at ({x}, {y}): {'success' if action_ctx.success else 'failed'}"]

                if action_ctx.before_state:
                    mouse_before = action_ctx.before_state.mouse_position
                    response_parts.append(f"Mouse was at: {mouse_before}")
                    response_parts.append(f"Active window: {action_ctx.before_state.active_window}")

                if action_ctx.after_state and action_ctx.before_state:
                    changed = self.vision_context.screen_changed(
                        action_ctx.before_state, action_ctx.after_state
                    )
                    response_parts.append(f"Screen changed: {changed}")

                response_parts.append(f"Duration: {action_ctx.duration_ms:.0f}ms")

                return {"content": [{"type": "text", "text": " | ".join(response_parts)}]}

            elif name == "double_click":
                x, y = arguments["x"], arguments["y"]

                # Use vision-aware execution
                action_ctx = self.vision_executor.double_click(x, y)

                response_parts = [f"Double-click at ({x}, {y}): {'success' if action_ctx.success else 'failed'}"]
                if action_ctx.before_state:
                    response_parts.append(f"Window: {action_ctx.before_state.active_window}")
                response_parts.append(f"Duration: {action_ctx.duration_ms:.0f}ms")

                return {"content": [{"type": "text", "text": " | ".join(response_parts)}]}

            elif name == "scroll":
                amount = arguments["amount"]
                x, y = arguments.get("x"), arguments.get("y")

                # Use vision-aware execution
                action_ctx = self.vision_executor.scroll(amount, x, y)

                response_parts = [f"Scroll {'up' if amount > 0 else 'down'} by {abs(amount)}: {'success' if action_ctx.success else 'failed'}"]
                if action_ctx.before_state and action_ctx.after_state:
                    changed = self.vision_context.screen_changed(
                        action_ctx.before_state, action_ctx.after_state
                    )
                    response_parts.append(f"Screen changed: {changed}")

                return {"content": [{"type": "text", "text": " | ".join(response_parts)}]}

            # Keyboard control - WITH VISION AWARENESS
            elif name == "type_text":
                text = arguments["text"]
                action_ctx = self.vision_executor.type_text(text)

                response_parts = [f"Typed '{text[:30]}{'...' if len(text) > 30 else ''}': {'success' if action_ctx.success else 'failed'}"]
                if action_ctx.before_state:
                    response_parts.append(f"In: {action_ctx.before_state.active_window}")
                return {"content": [{"type": "text", "text": " | ".join(response_parts)}]}

            elif name == "press_key":
                key = arguments["key"]
                action_ctx = self.vision_executor.press_key(key)

                response_parts = [f"Pressed '{key}': {'success' if action_ctx.success else 'failed'}"]
                if action_ctx.before_state:
                    response_parts.append(f"In: {action_ctx.before_state.active_window}")
                return {"content": [{"type": "text", "text": " | ".join(response_parts)}]}

            elif name == "hotkey":
                keys = arguments["keys"]
                action_ctx = self.vision_executor.hotkey(*keys)

                response_parts = [f"Hotkey {'+'.join(keys)}: {'success' if action_ctx.success else 'failed'}"]
                if action_ctx.before_state:
                    response_parts.append(f"In: {action_ctx.before_state.active_window}")
                return {"content": [{"type": "text", "text": " | ".join(response_parts)}]}

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
                from .skills import SkillContext
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

            # Ambient Intelligence
            elif name == "get_briefing":
                fmt = arguments.get("format", "text")
                briefing = self.ambient.get_briefing(format=fmt)
                return {"content": [{"type": "text", "text": briefing}]}

            elif name == "list_worlds":
                worlds = self.ambient.list_worlds()
                if worlds:
                    lines = [f"Configured worlds ({len(worlds)}):"]
                    for world in worlds:
                        status = "active" if world.is_active_now() else "inactive"
                        goals = len([g for g in world.goals if g.status.value == "active"])
                        entities = len(world.entities)
                        lines.append(f"- {world.name} [{status}] (ID: {world.id})")
                        lines.append(f"  {world.description}")
                        lines.append(f"  Goals: {goals}, Entities: {entities}, Keywords: {len(world.keywords)}")
                    result = "\n".join(lines)
                else:
                    result = "No worlds configured. Use create_world to add one."
                return {"content": [{"type": "text", "text": result}]}

            elif name == "create_world":
                world = self.ambient.create_world(
                    name=arguments["name"],
                    description=arguments["description"],
                    keywords=arguments.get("keywords", [])
                )
                return {"content": [{"type": "text", "text": f"Created world: {world.name} (ID: {world.id})"}]}

            elif name == "add_entity":
                entity = self.ambient.add_entity(
                    world_id=arguments["world_id"],
                    name=arguments["name"],
                    entity_type=arguments.get("entity_type", "custom"),
                    relationship=arguments.get("relationship", "watch"),
                    importance=arguments.get("importance", 0.5),
                    watch_for=arguments.get("watch_for", [])
                )
                if entity:
                    return {"content": [{"type": "text", "text": f"Added entity: {entity.name} (ID: {entity.id})"}]}
                return {"content": [{"type": "text", "text": "Failed to add entity. Check world_id."}]}

            elif name == "add_goal":
                goal = self.ambient.add_goal(
                    world_id=arguments["world_id"],
                    description=arguments["description"],
                    priority=arguments.get("priority", "medium"),
                    deadline=arguments.get("deadline")
                )
                if goal:
                    return {"content": [{"type": "text", "text": f"Added goal: {goal.description[:50]}... (ID: {goal.id})"}]}
                return {"content": [{"type": "text", "text": "Failed to add goal. Check world_id."}]}

            elif name == "get_insights":
                insights = self.ambient.get_pending_insights(
                    limit=arguments.get("limit", 5),
                    priority=arguments.get("priority")
                )
                if insights:
                    lines = [f"Pending insights ({len(insights)}):"]
                    for insight in insights:
                        lines.append(f"[{insight.priority.value}] {insight.title}")
                        if insight.suggested_action:
                            lines.append(f"  Action: {insight.suggested_action}")
                    result = "\n".join(lines)
                else:
                    result = "No pending insights."
                return {"content": [{"type": "text", "text": result}]}

            elif name == "get_ambient_status":
                status = self.ambient.get_status()
                lines = [
                    "Ambient Intelligence Status:",
                    f"  Running: {status.get('running', False)}",
                    f"  Worlds: {status.get('worlds', 0)}",
                    f"  Active worlds: {status.get('active_worlds', 0)}",
                ]
                if status.get('running'):
                    lines.extend([
                        f"  Cycle count: {status.get('cycle_count', 0)}",
                        f"  Total insights: {status.get('total_insights_generated', 0)}",
                        f"  Signals in cache: {status.get('signals_in_cache', 0)}",
                    ])
                return {"content": [{"type": "text", "text": "\n".join(lines)}]}

            elif name == "ambient_command":
                result = self._process_ambient_command(
                    arguments["command"],
                    arguments.get("context", "")
                )
                return {"content": [{"type": "text", "text": result}]}

            else:
                return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}]}

        except Exception as e:
            return {"content": [{"type": "text", "text": f"Error: {str(e)}"}]}

    def _process_ambient_command(self, command: str, context: str = "") -> str:
        """
        Process a natural language command related to ambient intelligence.

        This interprets conversational input and routes to the appropriate
        ambient system functions.

        Args:
            command: Natural language command from user
            context: Optional conversation context

        Returns:
            Natural language response
        """
        cmd_lower = command.lower().strip()

        # First, try the built-in voice command handler
        voice_response = self.ambient.handle_voice_command(cmd_lower)
        if voice_response:
            return voice_response

        # Pattern matching for world creation
        world_patterns = [
            ("i work in ", "work"),
            ("i'm in ", "work"),
            ("my business is ", "work"),
            ("i run a ", "work"),
            ("i have a ", "work"),
            ("my startup is ", "work"),
            ("create a world for ", None),
            ("add world ", None),
            ("new world ", None),
        ]

        for pattern, domain_type in world_patterns:
            if pattern in cmd_lower:
                # Extract the domain name
                idx = cmd_lower.index(pattern) + len(pattern)
                domain_name = command[idx:].strip()
                # Clean up common suffixes
                for suffix in [" business", " industry", " field", " sector", " work"]:
                    if domain_name.lower().endswith(suffix):
                        domain_name = domain_name[:-len(suffix)]

                if domain_name:
                    # Capitalize properly
                    domain_name = domain_name.title()
                    description = f"User's {domain_name.lower()} domain"
                    if domain_type:
                        description = f"User's {domain_type} in {domain_name.lower()}"

                    world = self.ambient.create_world(
                        name=domain_name,
                        description=description,
                        keywords=[domain_name.lower()]
                    )
                    return f"Created world '{world.name}'. I'll start monitoring this domain for you. You can add entities to track with 'track [company/person] in {world.name}' or goals with 'my goal is...'."

        # Pattern matching for entity tracking
        entity_patterns = [
            ("track ", None),
            ("monitor ", None),
            ("watch ", None),
            ("follow ", None),
            ("keep an eye on ", None),
        ]

        for pattern, _ in entity_patterns:
            if cmd_lower.startswith(pattern):
                rest = command[len(pattern):].strip()

                # Parse entity and optional world/relationship
                entity_name = rest
                relationship = "watch"
                world_id = None

                # Check for relationship indicators
                rel_indicators = {
                    " as a competitor": "competitor",
                    " as competitor": "competitor",
                    " competitor": "competitor",
                    " as a client": "client",
                    " as client": "client",
                    " as a prospect": "prospect",
                    " as prospect": "prospect",
                    " as a partner": "partner",
                    " as partner": "partner",
                }

                for indicator, rel in rel_indicators.items():
                    if indicator in rest.lower():
                        idx = rest.lower().index(indicator)
                        entity_name = rest[:idx].strip()
                        relationship = rel
                        break

                # Check for world specification
                world_indicators = [" in ", " for ", " under "]
                for indicator in world_indicators:
                    if indicator in entity_name.lower():
                        idx = entity_name.lower().index(indicator)
                        potential_world = entity_name[idx + len(indicator):].strip()
                        entity_name = entity_name[:idx].strip()
                        # Try to find the world
                        world = self.ambient.get_world_by_name(potential_world)
                        if world:
                            world_id = world.id
                        break

                # If no world specified, use the first world or prompt
                if not world_id:
                    worlds = self.ambient.list_worlds()
                    if worlds:
                        world_id = worlds[0].id
                    else:
                        return f"I'd like to track '{entity_name}', but you don't have any worlds set up yet. Try 'I work in [your industry]' first."

                # Determine entity type from name
                entity_type = "custom"
                if any(word in entity_name.lower() for word in ["inc", "corp", "llc", "company", "co."]):
                    entity_type = "company"

                entity = self.ambient.add_entity(
                    world_id=world_id,
                    name=entity_name,
                    entity_type=entity_type,
                    relationship=relationship,
                )

                if entity:
                    world = self.ambient.get_world(world_id)
                    world_name = world.name if world else "your world"
                    return f"Now tracking '{entity.name}' as {relationship} in {world_name}."
                return f"Couldn't add entity. Please check that the world exists."

        # Pattern matching for goals
        goal_patterns = [
            ("my goal is ", None),
            ("i want to ", None),
            ("i need to ", None),
            ("goal: ", None),
            ("add goal ", None),
        ]

        for pattern, _ in goal_patterns:
            if cmd_lower.startswith(pattern):
                goal_desc = command[len(pattern):].strip()

                # Check for priority indicators
                priority = "medium"
                priority_indicators = {
                    "urgently": "high",
                    "asap": "high",
                    "critical": "critical",
                    "important": "high",
                    "eventually": "low",
                    "someday": "low",
                }
                for indicator, pri in priority_indicators.items():
                    if indicator in goal_desc.lower():
                        priority = pri
                        break

                # Check for world specification
                world_id = None
                world_indicators = [" in ", " for "]
                for indicator in world_indicators:
                    if indicator in goal_desc.lower():
                        parts = goal_desc.lower().split(indicator)
                        if len(parts) > 1:
                            potential_world = parts[-1].strip()
                            world = self.ambient.get_world_by_name(potential_world)
                            if world:
                                world_id = world.id
                                goal_desc = goal_desc[:goal_desc.lower().rindex(indicator)].strip()
                            break

                # If no world specified, use the first world
                if not world_id:
                    worlds = self.ambient.list_worlds()
                    if worlds:
                        world_id = worlds[0].id
                    else:
                        return f"I'd like to add that goal, but you don't have any worlds set up yet. Try 'I work in [your industry]' first."

                goal = self.ambient.add_goal(
                    world_id=world_id,
                    description=goal_desc,
                    priority=priority,
                )

                if goal:
                    world = self.ambient.get_world(world_id)
                    world_name = world.name if world else "your world"
                    return f"Added goal '{goal_desc}' ({priority} priority) to {world_name}."
                return "Couldn't add goal. Please check that the world exists."

        # Pattern matching for adding keywords
        if "add keyword" in cmd_lower or "add keywords" in cmd_lower:
            # Extract keywords and world
            parts = cmd_lower.replace("add keyword", "").replace("add keywords", "").strip()
            worlds = self.ambient.list_worlds()
            if worlds:
                keywords = [k.strip() for k in parts.split(",") if k.strip()]
                if keywords:
                    for keyword in keywords:
                        self.ambient.add_keyword(worlds[0].id, keyword)
                    return f"Added keywords: {', '.join(keywords)}"
            return "No worlds to add keywords to."

        # Default: try to interpret as a general query
        # Return helpful guidance
        return f"""I didn't understand that ambient command. Here's what I can help with:

**Creating domains**: "I work in real estate", "my startup is in fintech"
**Tracking entities**: "track Compass as a competitor", "monitor Zillow"
**Setting goals**: "my goal is to close 3 deals this quarter"
**Getting updates**: "what's going on", "briefing", "status"
**Listing worlds**: "list my worlds", "what worlds do I have"

Your command was: "{command}"
"""

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
