"""
Aria Skills System

A flexible skill system inspired by:
- Superpowers: Markdown skills with YAML frontmatter
- Browser-Use: Decorator-based action registry
- Swarm: Simple function-based skills with handoffs

Skills can be defined in two ways:
1. Python functions with @skill decorator
2. Markdown files with YAML frontmatter
"""

from .base import Skill, SkillResult, SkillContext
from .registry import SkillRegistry, skill, get_registry
from .loader import SkillLoader, get_loader
from .hooks import HookManager, Hook, HookEvent, get_hooks, create_default_hooks

__all__ = [
    # Base classes
    "Skill",
    "SkillResult",
    "SkillContext",
    # Registry
    "SkillRegistry",
    "skill",
    "get_registry",
    # Loader
    "SkillLoader",
    "get_loader",
    # Hooks
    "HookManager",
    "Hook",
    "HookEvent",
    "get_hooks",
    "create_default_hooks",
]
