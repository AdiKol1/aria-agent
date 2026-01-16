"""
Base classes for specialist agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable

@dataclass
class AgentContext:
    """Context passed between agents."""
    user_input: str
    original_input: str = ""
    memory_context: str = ""
    screen_context: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.original_input:
            self.original_input = self.user_input

@dataclass
class AgentResult:
    """Result from agent execution."""
    success: bool
    response: str
    handoff_to: Optional[str] = None
    data: Optional[Dict] = None
    actions_taken: List[Dict] = field(default_factory=list)

    @classmethod
    def ok(cls, response: str, data: Dict = None) -> "AgentResult":
        """Create a successful result."""
        return cls(success=True, response=response, data=data)

    @classmethod
    def error(cls, response: str) -> "AgentResult":
        """Create a failure result."""
        return cls(success=False, response=response)

    @classmethod
    def handoff(cls, agent_name: str, response: str = "") -> "AgentResult":
        """Create a handoff result to another agent."""
        return cls(success=True, response=response, handoff_to=agent_name)

class BaseAgent(ABC):
    """Base class for specialist agents."""

    name: str = "base"
    description: str = "Base agent"
    triggers: List[str] = []

    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def register_tool(self, name: str, func: Callable):
        """Register a tool for this agent."""
        self.tools[name] = func

    @abstractmethod
    async def process(self, context: AgentContext) -> AgentResult:
        """Process a request. Must be implemented by subclasses."""
        pass

    def matches(self, text: str) -> float:
        """Return confidence score for handling this request."""
        text_lower = text.lower()
        score = 0.0

        for trigger in self.triggers:
            if trigger.lower() in text_lower:
                score = max(score, 0.8)

        # Check agent name
        if self.name.lower() in text_lower:
            score = max(score, 0.6)

        return score

    def can_handle(self, text: str, threshold: float = 0.5) -> bool:
        """Check if this agent can handle the request."""
        return self.matches(text) >= threshold
