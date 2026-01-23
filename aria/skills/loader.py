"""
Skill Loader for Aria.

Supports two skill formats:

1. **Anthropic Agent Skills format** (official, preferred):
   ```
   skill-name/
   ├── SKILL.md      # Required - frontmatter + instructions
   ├── scripts/      # Optional - executable code
   ├── references/   # Optional - additional docs
   └── assets/       # Optional - static resources
   ```

2. **Legacy flat format** (backward compatible):
   ```
   skill-name.md     # Single file with frontmatter
   ```

See https://agentskills.io/specification for the official spec.
"""

import os
import re
import yaml
import importlib
import importlib.util
from pathlib import Path
from typing import Optional, List, Dict, Any

from .base import Skill, SkillCategory, SkillContext, SkillResult
from .registry import SkillRegistry, get_registry


# Default paths
BUILTIN_SKILLS_PATH = Path(__file__).parent / "builtin"
USER_SKILLS_PATH = Path.home() / ".aria" / "skills"
USER_PYTHON_SKILLS_PATH = USER_SKILLS_PATH / "python"

# Name validation pattern (Anthropic spec)
SKILL_NAME_PATTERN = re.compile(r'^[a-z0-9]+(-[a-z0-9]+)*$')


class SkillLoader:
    """
    Loads and manages skills from various sources.

    Priority order (higher overrides lower):
    1. User Python skills (~/.aria/skills/python/*.py)
    2. User Markdown skills (~/.aria/skills/*.md)
    3. Built-in Python skills (aria/skills/builtin/*.py)
    4. Built-in Markdown skills (aria/skills/builtin/*.md)
    """

    def __init__(self, registry: Optional[SkillRegistry] = None):
        self.registry = registry or get_registry()
        self._loaded_files: List[str] = []

    def load_all(self) -> int:
        """
        Load all skills from all sources.
        Returns number of skills loaded.
        """
        count = 0

        # Ensure user skill directories exist
        USER_SKILLS_PATH.mkdir(parents=True, exist_ok=True)
        USER_PYTHON_SKILLS_PATH.mkdir(parents=True, exist_ok=True)

        # Load in priority order (lowest first, so higher priority can override)
        count += self._load_from_directory(BUILTIN_SKILLS_PATH, is_user=False)
        count += self._load_from_directory(USER_SKILLS_PATH, is_user=True)
        count += self._load_from_directory(USER_PYTHON_SKILLS_PATH, is_user=True)

        print(f"Loaded {count} skills ({self.registry.count()} active)")
        return count

    def _load_from_directory(self, path: Path, is_user: bool = False) -> int:
        """Load all skills from a directory."""
        if not path.exists():
            return 0

        count = 0

        # Load Anthropic-format skills (skill-name/SKILL.md)
        for skill_dir in path.iterdir():
            if skill_dir.is_dir() and not skill_dir.name.startswith(("_", ".")):
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    try:
                        skill = self._load_anthropic_skill(skill_dir, is_user)
                        if skill:
                            self.registry.register(skill)
                            self._loaded_files.append(str(skill_file))
                            count += 1
                    except Exception as e:
                        print(f"Error loading skill {skill_dir.name}: {e}")

        # Load legacy flat Markdown skills (*.md)
        for md_file in path.glob("*.md"):
            try:
                skill = self._load_markdown_skill(md_file, is_user)
                if skill:
                    self.registry.register(skill)
                    self._loaded_files.append(str(md_file))
                    count += 1
            except Exception as e:
                print(f"Error loading skill {md_file}: {e}")

        # Load Python skills
        for py_file in path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue  # Skip __init__.py etc
            try:
                loaded = self._load_python_skills(py_file, is_user)
                count += loaded
                if loaded > 0:
                    self._loaded_files.append(str(py_file))
            except Exception as e:
                print(f"Error loading skills from {py_file}: {e}")

        return count

    def _validate_skill_name(self, name: str) -> bool:
        """Validate skill name follows Anthropic spec."""
        if not name or len(name) > 64:
            return False
        return bool(SKILL_NAME_PATTERN.match(name))

    def _load_anthropic_skill(self, skill_dir: Path, is_user: bool = False) -> Optional[Skill]:
        """
        Load a skill in Anthropic Agent Skills format.

        Format:
        skill-name/
        ├── SKILL.md      # Required
        ├── scripts/      # Optional
        ├── references/   # Optional
        └── assets/       # Optional
        """
        skill_file = skill_dir / "SKILL.md"
        content = skill_file.read_text()

        # Parse YAML frontmatter
        if not content.startswith("---"):
            print(f"Skill {skill_dir.name} SKILL.md missing frontmatter")
            return None

        end_idx = content.find("---", 3)
        if end_idx == -1:
            print(f"Skill {skill_dir.name} SKILL.md has invalid frontmatter")
            return None

        frontmatter = content[3:end_idx].strip()
        instructions = content[end_idx + 3:].strip()

        try:
            meta = yaml.safe_load(frontmatter)
        except yaml.YAMLError as e:
            print(f"Invalid YAML in {skill_dir.name}/SKILL.md: {e}")
            return None

        # Validate required fields
        if "name" not in meta:
            print(f"Skill {skill_dir.name} missing name field")
            return None
        if "description" not in meta:
            print(f"Skill {skill_dir.name} missing description field")
            return None

        name = meta["name"]

        # Validate name format
        if not self._validate_skill_name(name):
            print(f"Skill name '{name}' doesn't follow spec (lowercase, hyphens only)")

        # Validate name matches directory
        if name != skill_dir.name:
            print(f"Warning: Skill name '{name}' doesn't match directory '{skill_dir.name}'")

        # Load resource paths
        scripts_dir = skill_dir / "scripts"
        references_dir = skill_dir / "references"
        assets_dir = skill_dir / "assets"

        # Build resource info
        resources = {}
        if scripts_dir.exists():
            resources["scripts"] = [str(f) for f in scripts_dir.iterdir() if f.is_file()]
        if references_dir.exists():
            resources["references"] = [str(f) for f in references_dir.iterdir() if f.is_file()]
        if assets_dir.exists():
            resources["assets"] = [str(f) for f in assets_dir.iterdir() if f.is_file()]

        # Parse category (Aria extension)
        category_str = meta.get("category", "custom")
        try:
            category = SkillCategory(category_str)
        except ValueError:
            category = SkillCategory.CUSTOM

        # Extract triggers from description if not provided (Aria extension)
        triggers = meta.get("triggers", [])
        if not triggers:
            # Generate triggers from name
            triggers = [name.replace("-", " ")]

        return Skill(
            name=name,
            description=meta["description"],
            triggers=triggers,
            category=category,
            instructions=instructions,
            requires_screen=meta.get("requires_screen", False),
            requires_confirmation=meta.get("requires_confirmation", False),
            is_destructive=meta.get("is_destructive", False),
            source_file=str(skill_file),
            is_user_skill=is_user,
            priority=meta.get("priority", 10 if is_user else 0),
        )

    def _load_markdown_skill(self, path: Path, is_user: bool = False) -> Optional[Skill]:
        """
        Load a skill from a markdown file with YAML frontmatter.

        Expected format:
        ---
        name: skill-name
        description: What the skill does
        triggers: ["keyword1", "keyword2"]
        category: system  # Optional: system, file, browser, memory, voice, workflow, custom
        requires_screen: false
        requires_confirmation: false
        ---
        # Skill Instructions

        Your instructions here...
        """
        content = path.read_text()

        # Parse YAML frontmatter
        if not content.startswith("---"):
            print(f"Skill {path} missing YAML frontmatter")
            return None

        # Find end of frontmatter
        end_idx = content.find("---", 3)
        if end_idx == -1:
            print(f"Skill {path} has invalid frontmatter")
            return None

        frontmatter = content[3:end_idx].strip()
        instructions = content[end_idx + 3:].strip()

        try:
            meta = yaml.safe_load(frontmatter)
        except yaml.YAMLError as e:
            print(f"Invalid YAML in {path}: {e}")
            return None

        # Validate required fields
        if "name" not in meta or "description" not in meta:
            print(f"Skill {path} missing name or description")
            return None

        # Parse category
        category_str = meta.get("category", "custom")
        try:
            category = SkillCategory(category_str)
        except ValueError:
            category = SkillCategory.CUSTOM

        return Skill(
            name=meta["name"],
            description=meta["description"],
            triggers=meta.get("triggers", []),
            category=category,
            instructions=instructions,
            requires_screen=meta.get("requires_screen", False),
            requires_confirmation=meta.get("requires_confirmation", False),
            is_destructive=meta.get("is_destructive", False),
            source_file=str(path),
            is_user_skill=is_user,
            priority=meta.get("priority", 10 if is_user else 0),
        )

    def _load_python_skills(self, path: Path, is_user: bool = False) -> int:
        """
        Load skills from a Python file.

        The file should use the @skill decorator to register skills.
        This function imports the module which triggers the decorators.
        """
        # Generate unique module name
        module_name = f"aria_skill_{path.stem}_{id(path)}"

        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return 0

        module = importlib.util.module_from_spec(spec)

        # Count skills before
        before = self.registry.count()

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"Error executing {path}: {e}")
            return 0

        # Count skills after
        after = self.registry.count()

        # Mark user skills
        if is_user and after > before:
            # Get newly registered skills and mark as user skills
            for skill in self.registry.all():
                if skill.source_file is None:
                    skill.source_file = str(path)
                    skill.is_user_skill = is_user

        return after - before

    def reload_user_skills(self) -> int:
        """
        Reload all user skills.
        Useful for hot-reloading after user edits their skills.
        """
        # Remove user skills
        for skill in list(self.registry.all()):
            if skill.is_user_skill:
                self.registry.unregister(skill.name)

        # Reload from user directories
        count = self._load_from_directory(USER_SKILLS_PATH, is_user=True)
        count += self._load_from_directory(USER_PYTHON_SKILLS_PATH, is_user=True)

        print(f"Reloaded {count} user skills")
        return count

    def create_user_skill_template(self, name: str, use_anthropic_format: bool = True) -> Path:
        """
        Create a template skill for the user to customize.

        Args:
            name: Skill name (lowercase, hyphens only)
            use_anthropic_format: If True, creates folder with SKILL.md (official format)
                                  If False, creates flat name.md file (legacy format)

        Returns:
            Path to the created SKILL.md or name.md file
        """
        # Validate and normalize name
        name = name.lower().replace(" ", "-").replace("_", "-")
        if not self._validate_skill_name(name):
            # Try to fix common issues
            name = re.sub(r'[^a-z0-9-]', '', name)
            name = re.sub(r'-+', '-', name).strip('-')

        if use_anthropic_format:
            # Create Anthropic-format skill folder
            skill_dir = USER_SKILLS_PATH / name
            skill_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories
            (skill_dir / "scripts").mkdir(exist_ok=True)
            (skill_dir / "references").mkdir(exist_ok=True)

            template = f'''---
name: {name}
description: Replace with description of the skill and when Aria should use it. Include keywords that help identify relevant tasks.
---

# {name.replace("-", " ").title()}

## Instructions

[Your instructions here - this is what Claude/Aria will follow when executing this skill]

## Examples

- Example usage 1
- Example usage 2

## Guidelines

- Guideline 1
- Guideline 2
'''
            path = skill_dir / "SKILL.md"
            path.write_text(template)
            print(f"Created Anthropic-format skill: {skill_dir}/")
            print(f"  - SKILL.md (edit this)")
            print(f"  - scripts/ (add executable scripts)")
            print(f"  - references/ (add reference docs)")
        else:
            # Create legacy flat file
            template = f'''---
name: {name}
description: Description of what this skill does
triggers: ["{name.replace("-", " ")}", "custom trigger"]
category: custom
---

# {name.replace("-", " ").title()}

## When to Use
Describe when Aria should use this skill.

## Instructions
1. Step one
2. Step two
3. Step three

## Example
User: "example request"
Aria: "example response"
'''
            path = USER_SKILLS_PATH / f"{name}.md"
            path.write_text(template)
            print(f"Created skill template: {path}")

        return path

    def get_loaded_files(self) -> List[str]:
        """Get list of files that skills were loaded from."""
        return self._loaded_files.copy()


# Singleton instance
_loader: Optional[SkillLoader] = None


def get_loader() -> SkillLoader:
    """Get the global skill loader."""
    global _loader
    if _loader is None:
        _loader = SkillLoader()
    return _loader
