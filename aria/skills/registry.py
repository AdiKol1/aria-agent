"""
Skill Registry for Aria.

Provides:
- @skill decorator for registering Python functions as skills
- SkillRegistry singleton for managing all skills
- Skill lookup by name, category, and trigger matching
"""

from typing import Callable, Optional, List, Dict, Any
from functools import wraps
import asyncio

from .base import Skill, SkillCategory, SkillContext, SkillResult


class SkillRegistry:
    """
    Central registry for all Aria skills.

    Skills are stored by name and can be looked up by:
    - Exact name match
    - Category filter
    - Trigger keyword matching
    - Best match for natural language input
    """

    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._by_category: Dict[SkillCategory, List[str]] = {
            cat: [] for cat in SkillCategory
        }

    def register(self, skill: Skill) -> None:
        """Register a skill in the registry."""
        # User skills can shadow system skills
        if skill.name in self._skills:
            existing = self._skills[skill.name]
            if skill.is_user_skill and not existing.is_user_skill:
                # User skill shadows system skill
                print(f"User skill '{skill.name}' shadows built-in skill")
            elif not skill.is_user_skill and existing.is_user_skill:
                # Don't override user skill with system skill
                return

        self._skills[skill.name] = skill

        # Index by category
        if skill.name not in self._by_category[skill.category]:
            self._by_category[skill.category].append(skill.name)

    def unregister(self, name: str) -> bool:
        """Remove a skill from the registry."""
        if name in self._skills:
            skill = self._skills[name]
            del self._skills[name]
            if name in self._by_category[skill.category]:
                self._by_category[skill.category].remove(name)
            return True
        return False

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by exact name."""
        return self._skills.get(name)

    def get_by_category(self, category: SkillCategory) -> List[Skill]:
        """Get all skills in a category."""
        return [self._skills[name] for name in self._by_category[category]]

    def find_matching(self, text: str, min_score: float = 0.3) -> List[tuple[Skill, float]]:
        """
        Find skills matching the given text.
        Returns list of (skill, score) tuples, sorted by score descending.
        """
        matches = []
        for skill in self._skills.values():
            score = skill.matches(text)
            if score >= min_score:
                matches.append((skill, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def best_match(self, text: str) -> Optional[Skill]:
        """Get the best matching skill for the given text."""
        matches = self.find_matching(text)
        return matches[0][0] if matches else None

    def all(self) -> List[Skill]:
        """Get all registered skills."""
        return list(self._skills.values())

    def all_names(self) -> List[str]:
        """Get names of all registered skills."""
        return list(self._skills.keys())

    def count(self) -> int:
        """Get the number of registered skills."""
        return len(self._skills)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get Claude tool schemas for all skills."""
        return [skill.to_tool_schema() for skill in self._skills.values()]

    def describe_all(self) -> str:
        """Get a formatted description of all skills for prompts."""
        lines = ["Available Skills:\n"]
        for category in SkillCategory:
            skills = self.get_by_category(category)
            if skills:
                lines.append(f"\n## {category.value.title()}")
                for skill in skills:
                    triggers = ", ".join(skill.triggers[:3]) if skill.triggers else "no specific triggers"
                    lines.append(f"- **{skill.name}**: {skill.description}")
                    lines.append(f"  Triggers: {triggers}")
        return "\n".join(lines)


# Singleton instance
_registry: Optional[SkillRegistry] = None


def get_registry() -> SkillRegistry:
    """Get the global skill registry."""
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry


def skill(
    name: str,
    description: str,
    triggers: Optional[List[str]] = None,
    category: SkillCategory = SkillCategory.CUSTOM,
    requires_screen: bool = False,
    requires_confirmation: bool = False,
    is_destructive: bool = False,
):
    """
    Decorator to register a function as a skill.

    Usage:
        @skill(
            name="open_app",
            description="Open an application",
            triggers=["open", "launch", "start"],
            category=SkillCategory.SYSTEM
        )
        def open_app(context: SkillContext) -> SkillResult:
            # ... implementation
            return SkillResult.ok("Opened app")

        # Or for async functions:
        @skill(...)
        async def open_app(context: SkillContext) -> SkillResult:
            # ... async implementation
            return SkillResult.ok("Opened app")
    """
    def decorator(func: Callable[[SkillContext], SkillResult]):
        # Wrap sync functions to be async-compatible
        @wraps(func)
        async def async_wrapper(context: SkillContext) -> SkillResult:
            result = func(context)
            if asyncio.iscoroutine(result):
                return await result
            return result

        # Create and register the skill
        skill_obj = Skill(
            name=name,
            description=description,
            triggers=triggers or [],
            category=category,
            handler=async_wrapper,
            requires_screen=requires_screen,
            requires_confirmation=requires_confirmation,
            is_destructive=is_destructive,
        )

        registry = get_registry()
        registry.register(skill_obj)

        # Return original function for direct calls
        return func

    return decorator
