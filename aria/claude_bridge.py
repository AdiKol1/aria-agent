"""
Claude Code Bridge

Allows Aria to delegate coding tasks to Claude Code CLI.
Enables voice-controlled coding: "Hey Aria, fix the bug in login.ts"
"""

import os
import subprocess
import threading
from pathlib import Path
from typing import Optional, Callable

from .memory import get_memory


class ClaudeBridge:
    """Bridge between Aria and Claude Code CLI."""

    def __init__(self):
        self.memory = get_memory()
        self.current_project: Optional[Path] = None
        self._detect_project()

    def _detect_project(self):
        """Try to detect the current project directory."""
        # Check common locations
        home = Path.home()

        # Check if we're in a git repo
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                cwd=home / "Desktop"
            )
            if result.returncode == 0:
                self.current_project = Path(result.stdout.strip())
        except:
            pass

        # Default to Desktop if no project detected
        if not self.current_project:
            self.current_project = home / "Desktop"

    def set_project(self, path: str) -> bool:
        """Set the current project directory."""
        project_path = Path(path).expanduser()
        if project_path.exists() and project_path.is_dir():
            self.current_project = project_path
            self.memory.remember_fact(
                f"Current project is at {project_path}",
                "project"
            )
            return True
        return False

    def is_coding_request(self, user_input: str) -> bool:
        """Determine if a request should be handled by Claude Code."""
        coding_keywords = [
            # Direct coding actions
            "code", "coding", "program", "debug", "fix bug", "write code",
            "implement", "refactor", "create function", "add feature",

            # File operations
            "edit file", "modify file", "update file", "create file",
            "read file", "show file", "find file", "search code",

            # Git operations
            "commit", "push", "pull", "branch", "merge", "git",

            # Project operations
            "run tests", "build", "deploy", "install dependencies",
            "npm", "pip", "yarn", "cargo",

            # Specific coding phrases
            "in the codebase", "in the code", "in the repo",
            "this function", "this class", "this component",
            "fix the", "debug the", "update the", "refactor the",

            # Claude Code specific
            "claude code", "use claude", "ask claude code"
        ]

        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in coding_keywords)

    def run_claude(
        self,
        prompt: str,
        on_output: Optional[Callable[[str], None]] = None,
        timeout: int = 300
    ) -> str:
        """
        Run Claude Code CLI with a prompt.

        Args:
            prompt: The prompt to send to Claude Code
            on_output: Optional callback for streaming output
            timeout: Timeout in seconds (default 5 minutes)

        Returns:
            Claude Code's response
        """
        try:
            # Build the command
            cmd = [
                "claude",
                "--print",  # Print output without interactive mode
                prompt
            ]

            # Run in the project directory
            cwd = str(self.current_project) if self.current_project else None

            print(f"Running Claude Code in {cwd}: {prompt[:50]}...")

            # Run the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=timeout,
                env={**os.environ, "CLAUDE_CODE_NO_COLOR": "1"}
            )

            output = result.stdout
            if result.stderr:
                output += f"\n\nErrors:\n{result.stderr}"

            # Store this interaction
            self.memory.remember_interaction(
                f"Coding task: {prompt[:100]}",
                prompt,
                "completed" if result.returncode == 0 else "failed"
            )

            return output

        except subprocess.TimeoutExpired:
            return "Claude Code timed out. The task may be too complex or Claude Code may be waiting for input."
        except FileNotFoundError:
            return "Claude Code CLI not found. Make sure 'claude' is installed and in your PATH."
        except Exception as e:
            return f"Error running Claude Code: {str(e)}"

    def run_claude_async(
        self,
        prompt: str,
        on_complete: Callable[[str], None],
        on_status: Optional[Callable[[str], None]] = None
    ):
        """
        Run Claude Code asynchronously in a background thread.

        Args:
            prompt: The prompt to send
            on_complete: Callback with the result when done
            on_status: Optional callback for status updates
        """
        def _run():
            if on_status:
                on_status("Starting Claude Code...")

            result = self.run_claude(prompt)

            if on_status:
                on_status("Claude Code finished")

            on_complete(result)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def summarize_for_voice(self, claude_output: str, max_length: int = 200) -> str:
        """
        Summarize Claude Code output for voice response.

        Args:
            claude_output: Full output from Claude Code
            max_length: Maximum characters for voice response

        Returns:
            Concise summary suitable for speaking
        """
        # Remove ANSI codes and extra whitespace
        import re
        clean = re.sub(r'\x1b\[[0-9;]*m', '', claude_output)
        clean = re.sub(r'\s+', ' ', clean).strip()

        # Look for key indicators
        if "error" in clean.lower():
            # Extract error summary
            lines = clean.split('\n')
            for line in lines:
                if "error" in line.lower():
                    return f"There was an error: {line[:100]}"
            return "There was an error. Check the terminal for details."

        if "success" in clean.lower() or "done" in clean.lower() or "completed" in clean.lower():
            return "Done! The task completed successfully."

        if "created" in clean.lower():
            return "Created the requested files."

        if "modified" in clean.lower() or "updated" in clean.lower():
            return "Updated the files as requested."

        # Default: first meaningful sentence
        sentences = clean.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Skip very short fragments
                return sentence[:max_length] + ("..." if len(sentence) > max_length else "")

        return "Task completed. Check the output for details."


# Singleton
_bridge: Optional[ClaudeBridge] = None


def get_claude_bridge() -> ClaudeBridge:
    """Get the singleton ClaudeBridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = ClaudeBridge()
    return _bridge
