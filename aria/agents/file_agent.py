"""
File operations specialist agent.
"""

from .base import BaseAgent, AgentContext, AgentResult
import os
from pathlib import Path

class FileAgent(BaseAgent):
    """Handles file operations: find, read, write, move, delete."""

    name = "file"
    description = "Handles file operations like finding, reading, writing, moving, and deleting files"
    triggers = [
        "file", "find file", "search file", "locate",
        "read file", "open file", "show file",
        "create file", "write file", "save file",
        "move file", "copy file", "rename",
        "delete file", "remove file",
        "list files", "show files", "directory"
    ]

    async def process(self, context: AgentContext) -> AgentResult:
        """Process file operation request."""
        input_lower = context.user_input.lower()

        # Determine operation type
        if any(w in input_lower for w in ["find", "search", "locate", "where"]):
            return await self._find_file(context)
        elif any(w in input_lower for w in ["read", "open", "show", "cat", "view"]):
            return await self._read_file(context)
        elif any(w in input_lower for w in ["create", "write", "save", "new"]):
            return await self._create_file(context)
        elif any(w in input_lower for w in ["move", "rename", "mv"]):
            return await self._move_file(context)
        elif any(w in input_lower for w in ["copy", "cp", "duplicate"]):
            return await self._copy_file(context)
        elif any(w in input_lower for w in ["delete", "remove", "rm"]):
            return await self._delete_file(context)
        elif any(w in input_lower for w in ["list", "ls", "directory", "dir"]):
            return await self._list_files(context)
        else:
            return AgentResult.error("I couldn't determine the file operation you want.")

    async def _find_file(self, context: AgentContext) -> AgentResult:
        """Find files matching a pattern."""
        # Extract filename pattern from input
        # This is a simplified implementation
        import subprocess

        # Try to extract what to search for
        words = context.user_input.lower().split()
        search_term = None
        for i, word in enumerate(words):
            if word in ["find", "search", "locate", "for"]:
                if i + 1 < len(words):
                    search_term = words[i + 1]
                    break

        if not search_term:
            return AgentResult.error("What file should I search for?")

        try:
            result = subprocess.run(
                ["find", str(Path.home()), "-name", f"*{search_term}*", "-type", "f"],
                capture_output=True,
                text=True,
                timeout=10
            )
            files = result.stdout.strip().split("\n")[:10]  # Limit results

            if files and files[0]:
                return AgentResult.ok(
                    f"Found {len(files)} files:\n" + "\n".join(files),
                    data={"files": files}
                )
            else:
                return AgentResult.ok(f"No files found matching '{search_term}'")
        except Exception as e:
            return AgentResult.error(f"Search failed: {e}")

    async def _read_file(self, context: AgentContext) -> AgentResult:
        """Read a file's contents."""
        # Extract file path
        words = context.user_input.split()
        file_path = None

        for word in words:
            if "/" in word or word.endswith((".txt", ".py", ".md", ".json")):
                file_path = word
                break

        if not file_path:
            return AgentResult.error("Which file should I read?")

        path = Path(file_path).expanduser()

        if not path.exists():
            return AgentResult.error(f"File not found: {path}")

        try:
            content = path.read_text()
            # Truncate if too long
            if len(content) > 2000:
                content = content[:2000] + "\n... (truncated)"
            return AgentResult.ok(f"Contents of {path.name}:\n{content}")
        except Exception as e:
            return AgentResult.error(f"Failed to read file: {e}")

    async def _create_file(self, context: AgentContext) -> AgentResult:
        """Create a new file."""
        return AgentResult.error("Creating files requires more specific instructions. Please specify the file path and content.")

    async def _move_file(self, context: AgentContext) -> AgentResult:
        """Move or rename a file."""
        return AgentResult.error("Moving files requires source and destination paths.")

    async def _copy_file(self, context: AgentContext) -> AgentResult:
        """Copy a file."""
        return AgentResult.error("Copying files requires source and destination paths.")

    async def _delete_file(self, context: AgentContext) -> AgentResult:
        """Delete a file (with confirmation)."""
        return AgentResult(
            success=True,
            response="Deleting files is a destructive operation. Please confirm the exact file path.",
            data={"requires_confirmation": True}
        )

    async def _list_files(self, context: AgentContext) -> AgentResult:
        """List files in a directory."""
        # Default to current directory or home
        path = Path.cwd()

        words = context.user_input.split()
        for word in words:
            if "/" in word or word.startswith("~"):
                path = Path(word).expanduser()
                break

        if not path.exists():
            return AgentResult.error(f"Directory not found: {path}")

        try:
            files = list(path.iterdir())[:20]  # Limit results
            file_list = "\n".join([
                f"{'📁' if f.is_dir() else '📄'} {f.name}"
                for f in sorted(files)
            ])
            return AgentResult.ok(f"Contents of {path}:\n{file_list}")
        except Exception as e:
            return AgentResult.error(f"Failed to list directory: {e}")
