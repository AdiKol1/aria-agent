"""
Multi-Agent System for Aria.

Specialist agents with Swarm-style handoffs.
"""

from .base import BaseAgent, AgentContext, AgentResult
from .coordinator import Coordinator, get_coordinator
from .file_agent import FileAgent
from .browser_agent import BrowserAgent
from .system_agent import SystemAgent
from .code_agent import CodeAgent

__all__ = [
    "BaseAgent", "AgentContext", "AgentResult",
    "Coordinator", "get_coordinator",
    "FileAgent", "BrowserAgent", "SystemAgent", "CodeAgent",
]
