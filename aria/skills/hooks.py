"""
Hook System for Aria.

Provides lifecycle hooks that can inject context or perform actions
at key moments in the agent's operation.

Inspired by the Superpowers pattern, hooks allow:
- Injecting context on session start
- Performing actions before/after each request
- Cleaning up on session end

Hook types:
- SessionStart: When Aria wakes up or session begins
- SessionEnd: When session ends
- BeforeRequest: Before processing user input
- AfterRequest: After generating response
- BeforeAction: Before executing an action
- AfterAction: After executing an action
"""

import json
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any
import yaml


class HookEvent(Enum):
    """Events that can trigger hooks."""
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    BEFORE_REQUEST = "before_request"
    AFTER_REQUEST = "after_request"
    BEFORE_ACTION = "before_action"
    AFTER_ACTION = "after_action"
    ON_ERROR = "on_error"
    ON_MEMORY_RECALL = "on_memory_recall"


@dataclass
class HookContext:
    """Context passed to hooks."""
    event: HookEvent
    user_input: Optional[str] = None
    response: Optional[str] = None
    action: Optional[str] = None
    action_result: Optional[Any] = None
    error: Optional[str] = None
    memory_results: Optional[List[str]] = None
    variables: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HookResult:
    """Result from a hook execution."""
    success: bool = True
    context_injection: Optional[str] = None  # Text to inject into context
    should_continue: bool = True              # If False, abort current operation
    variables: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class Hook:
    """
    A hook that can be triggered by events.

    Hooks can be:
    - Python functions
    - Shell commands
    """
    name: str
    event: HookEvent
    handler: Optional[Callable[[HookContext], HookResult]] = None
    command: Optional[str] = None  # Shell command to run
    priority: int = 0              # Higher priority runs first
    enabled: bool = True

    def execute(self, context: HookContext) -> HookResult:
        """Execute the hook."""
        if not self.enabled:
            return HookResult(success=True)

        try:
            if self.handler is not None:
                return self.handler(context)
            elif self.command is not None:
                return self._run_command(context)
            return HookResult(success=True)
        except Exception as e:
            return HookResult(success=False, error=str(e))

    def _run_command(self, context: HookContext) -> HookResult:
        """Execute a shell command hook."""
        # Set up environment with context
        env = {
            "ARIA_EVENT": context.event.value,
            "ARIA_USER_INPUT": context.user_input or "",
            "ARIA_RESPONSE": context.response or "",
        }

        try:
            result = subprocess.run(
                self.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=5,
                env={**dict(subprocess.os.environ), **env}
            )

            if result.returncode == 0:
                # Command output becomes context injection
                return HookResult(
                    success=True,
                    context_injection=result.stdout.strip() if result.stdout else None
                )
            else:
                return HookResult(
                    success=False,
                    error=result.stderr or f"Command failed with code {result.returncode}"
                )
        except subprocess.TimeoutExpired:
            return HookResult(success=False, error="Hook command timed out")
        except Exception as e:
            return HookResult(success=False, error=str(e))


class HookManager:
    """
    Manages all hooks for Aria.

    Hooks can be registered programmatically or loaded from config.
    """

    def __init__(self):
        self._hooks: Dict[HookEvent, List[Hook]] = {
            event: [] for event in HookEvent
        }

    def register(self, hook: Hook) -> None:
        """Register a hook."""
        self._hooks[hook.event].append(hook)
        # Sort by priority (highest first)
        self._hooks[hook.event].sort(key=lambda h: h.priority, reverse=True)

    def register_function(
        self,
        name: str,
        event: HookEvent,
        handler: Callable[[HookContext], HookResult],
        priority: int = 0
    ) -> None:
        """Register a function as a hook."""
        hook = Hook(name=name, event=event, handler=handler, priority=priority)
        self.register(hook)

    def register_command(
        self,
        name: str,
        event: HookEvent,
        command: str,
        priority: int = 0
    ) -> None:
        """Register a shell command as a hook."""
        hook = Hook(name=name, event=event, command=command, priority=priority)
        self.register(hook)

    def unregister(self, name: str) -> bool:
        """Remove a hook by name."""
        for event in HookEvent:
            for hook in self._hooks[event]:
                if hook.name == name:
                    self._hooks[event].remove(hook)
                    return True
        return False

    def get_hooks(self, event: HookEvent) -> List[Hook]:
        """Get all hooks for an event."""
        return [h for h in self._hooks[event] if h.enabled]

    def trigger(self, context: HookContext) -> HookResult:
        """
        Trigger all hooks for the context's event.

        Returns combined result from all hooks.
        """
        combined = HookResult(success=True)
        context_injections = []

        for hook in self.get_hooks(context.event):
            result = hook.execute(context)

            if not result.success:
                print(f"Hook {hook.name} failed: {result.error}")
                # Continue with other hooks unless hook says to stop
                if not result.should_continue:
                    return result

            if result.context_injection:
                context_injections.append(result.context_injection)

            # Merge variables
            combined.variables.update(result.variables)

        # Combine all context injections
        if context_injections:
            combined.context_injection = "\n\n".join(context_injections)

        return combined

    def trigger_session_start(self) -> HookResult:
        """Convenience method for session start."""
        context = HookContext(event=HookEvent.SESSION_START)
        return self.trigger(context)

    def trigger_session_end(self) -> HookResult:
        """Convenience method for session end."""
        context = HookContext(event=HookEvent.SESSION_END)
        return self.trigger(context)

    def trigger_before_request(self, user_input: str) -> HookResult:
        """Convenience method for before request."""
        context = HookContext(
            event=HookEvent.BEFORE_REQUEST,
            user_input=user_input
        )
        return self.trigger(context)

    def trigger_after_request(self, user_input: str, response: str) -> HookResult:
        """Convenience method for after request."""
        context = HookContext(
            event=HookEvent.AFTER_REQUEST,
            user_input=user_input,
            response=response
        )
        return self.trigger(context)

    def load_from_config(self, config_path: Path) -> int:
        """
        Load hooks from a YAML config file.

        Expected format:
        hooks:
          session_start:
            - name: inject-context
              command: echo "Today is $(date)"
              priority: 10
          before_request:
            - name: log-request
              command: echo "$ARIA_USER_INPUT" >> ~/.aria/log.txt
        """
        if not config_path.exists():
            return 0

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading hook config: {e}")
            return 0

        if not config or "hooks" not in config:
            return 0

        count = 0
        for event_name, hooks in config["hooks"].items():
            try:
                event = HookEvent(event_name)
            except ValueError:
                print(f"Unknown hook event: {event_name}")
                continue

            for hook_config in hooks:
                if "name" not in hook_config:
                    continue

                hook = Hook(
                    name=hook_config["name"],
                    event=event,
                    command=hook_config.get("command"),
                    priority=hook_config.get("priority", 0),
                    enabled=hook_config.get("enabled", True)
                )
                self.register(hook)
                count += 1

        print(f"Loaded {count} hooks from config")
        return count

    def count(self) -> int:
        """Get total number of registered hooks."""
        return sum(len(hooks) for hooks in self._hooks.values())


# Singleton instance
_hooks: Optional[HookManager] = None


def get_hooks() -> HookManager:
    """Get the global hook manager."""
    global _hooks
    if _hooks is None:
        _hooks = HookManager()
    return _hooks


# Built-in hooks that can be used by default
def create_default_hooks(hook_manager: HookManager) -> None:
    """Register default hooks."""

    def session_start_hook(context: HookContext) -> HookResult:
        """Inject useful context at session start."""
        from datetime import datetime
        now = datetime.now()
        injection = f"""Current time: {now.strftime('%Y-%m-%d %H:%M')}
Day of week: {now.strftime('%A')}"""
        return HookResult(success=True, context_injection=injection)

    hook_manager.register_function(
        name="default-session-start",
        event=HookEvent.SESSION_START,
        handler=session_start_hook,
        priority=-10  # Low priority, let user hooks override
    )
