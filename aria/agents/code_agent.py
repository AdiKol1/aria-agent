"""
Coding specialist agent - delegates to Claude Code.
"""

from .base import BaseAgent, AgentContext, AgentResult

class CodeAgent(BaseAgent):
    """Handles coding tasks by delegating to Claude Code."""

    name = "code"
    description = "Handles coding tasks like writing code, fixing bugs, running tests, and git operations"
    triggers = [
        "code", "coding", "program", "implement",
        "fix bug", "fix the bug", "debug", "error",
        "test", "tests", "unittest",
        "commit", "push", "pull", "git",
        "refactor", "optimize", "improve code",
        "write function", "create class",
        "in the codebase", "in the code",
        ".py", ".js", ".ts", ".tsx", ".jsx",
        "function", "class", "variable"
    ]

    def __init__(self):
        super().__init__()
        try:
            from aria.claude_bridge import get_claude_bridge
            self.bridge = get_claude_bridge()
        except ImportError:
            self.bridge = None

    async def process(self, context: AgentContext) -> AgentResult:
        """Process coding request by delegating to Claude Code."""
        if not self.bridge:
            return AgentResult.error("Claude Code bridge not available")

        try:
            # Run Claude Code
            print(f"[CodeAgent] Delegating to Claude Code: {context.user_input[:50]}...")
            output = self.bridge.run_claude(context.user_input)

            # Summarize for voice
            summary = self.bridge.summarize_for_voice(output)

            return AgentResult.ok(
                summary,
                data={"full_output": output}
            )
        except Exception as e:
            return AgentResult.error(f"Claude Code error: {e}")
