"""
Base classes for the Aria Skills System.

Defines the core data structures:
- Skill: A capability Aria can invoke
- SkillResult: Output from skill execution
- SkillContext: Runtime context passed to skills
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, List, Dict
from enum import Enum


class SkillCategory(Enum):
    """Categories for organizing skills."""
    SYSTEM = "system"       # Desktop control, app management
    FILE = "file"           # File operations
    BROWSER = "browser"     # Web browsing and automation
    MEMORY = "memory"       # Memory and knowledge operations
    VOICE = "voice"         # Voice and TTS operations
    WORKFLOW = "workflow"   # Multi-step procedures
    CUSTOM = "custom"       # User-defined skills


@dataclass
class SkillContext:
    """
    Runtime context passed to skill handlers.

    Contains information about the current state that skills
    might need to execute properly.
    """
    user_input: str                          # Original user request
    screen_base64: Optional[str] = None      # Current screenshot
    memory_context: Optional[str] = None     # Relevant memory
    active_app: Optional[str] = None         # Frontmost application
    variables: Dict[str, Any] = field(default_factory=dict)  # Swarm-style context vars

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self.variables.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a context variable."""
        self.variables[key] = value


@dataclass
class SkillResult:
    """
    Result from executing a skill.

    Skills can return:
    - success: Whether the skill completed successfully
    - output: Text output to show/speak to user
    - data: Structured data for programmatic use
    - handoff_to: Another skill to invoke (Swarm pattern)
    - needs_confirmation: If True, ask user before proceeding
    """
    success: bool = True
    output: str = ""
    data: Optional[Dict[str, Any]] = None
    handoff_to: Optional[str] = None         # Skill name to hand off to
    needs_confirmation: bool = False
    confirmation_prompt: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def ok(cls, output: str = "", data: Optional[Dict] = None) -> "SkillResult":
        """Create a successful result."""
        return cls(success=True, output=output, data=data)

    @classmethod
    def fail(cls, error: str) -> "SkillResult":
        """Create a failed result."""
        return cls(success=False, error=error, output=f"Error: {error}")

    @classmethod
    def confirm(cls, prompt: str) -> "SkillResult":
        """Request confirmation before proceeding."""
        return cls(
            success=True,
            needs_confirmation=True,
            confirmation_prompt=prompt
        )

    @classmethod
    def handoff(cls, skill_name: str, output: str = "") -> "SkillResult":
        """Hand off to another skill."""
        return cls(success=True, output=output, handoff_to=skill_name)


@dataclass
class Skill:
    """
    A skill that Aria can invoke.

    Skills can be defined in two ways:
    1. Python function with @skill decorator (handler is set)
    2. Markdown file with instructions (instructions is set)

    For markdown skills, the instructions are passed to Claude
    as additional system prompt context.
    """
    name: str
    description: str
    triggers: List[str] = field(default_factory=list)  # Keywords that trigger this skill
    category: SkillCategory = SkillCategory.CUSTOM

    # For Python skills
    handler: Optional[Callable[[SkillContext], SkillResult]] = None

    # For Markdown skills
    instructions: Optional[str] = None

    # Behavior flags
    requires_screen: bool = False          # Needs screenshot to execute
    requires_confirmation: bool = False    # Always ask before executing
    is_destructive: bool = False           # Modifies system state

    # Metadata
    source_file: Optional[str] = None      # Path to skill file
    is_user_skill: bool = False            # User-defined vs built-in
    priority: int = 0                       # Higher = preferred when matching

    def matches(self, text: str) -> float:
        """
        Check if this skill matches the given text.
        Returns a score from 0.0 to 1.0.
        """
        text_lower = text.lower()

        # Check triggers
        for trigger in self.triggers:
            if trigger.lower() in text_lower:
                return 0.9 + (0.1 if self.is_user_skill else 0)

        # Check name
        if self.name.lower() in text_lower:
            return 0.7

        # Check description keywords
        desc_words = self.description.lower().split()
        matches = sum(1 for w in desc_words if w in text_lower and len(w) > 3)
        if matches > 0:
            return min(0.5, matches * 0.1)

        return 0.0

    def is_python_skill(self) -> bool:
        """Check if this is a Python-based skill."""
        return self.handler is not None

    def is_markdown_skill(self) -> bool:
        """Check if this is a Markdown-based skill."""
        return self.instructions is not None

    async def execute(self, context: SkillContext) -> SkillResult:
        """
        Execute the skill with the given context.

        For Python skills, calls the handler directly.
        For Markdown skills, returns instructions for Claude to follow.
        """
        if self.handler is not None:
            try:
                result = self.handler(context)
                # Support both sync and async handlers
                if hasattr(result, '__await__'):
                    result = await result
                return result
            except Exception as e:
                return SkillResult.fail(str(e))

        elif self.instructions is not None:
            # Return instructions for Claude to interpret
            return SkillResult.ok(
                output=f"Following skill: {self.name}",
                data={"instructions": self.instructions}
            )

        return SkillResult.fail(f"Skill {self.name} has no handler or instructions")

    def to_tool_schema(self) -> Dict[str, Any]:
        """
        Convert skill to Claude tool schema for function calling.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "The user's request or input for this skill"
                    }
                },
                "required": ["input"]
            }
        }

    def __repr__(self) -> str:
        skill_type = "python" if self.is_python_skill() else "markdown"
        return f"Skill({self.name}, type={skill_type}, category={self.category.value})"
