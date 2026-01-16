"""
Aria Task Planner

Decomposes goals into executable steps with verification and fallbacks.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .config import ANTHROPIC_API_KEY, CLAUDE_MODEL_FAST
from .lazy_anthropic import get_client as get_anthropic_client
from .intent import Intent


@dataclass
class PlanStep:
    """A single step in a plan."""
    action: Dict[str, Any]  # Action to execute
    description: str  # Human-readable description
    verification: str  # What to check after action
    fallback: Optional[Dict[str, Any]] = None  # Alternative if step fails


@dataclass
class Plan:
    """Execution plan for a task."""
    goal: str
    steps: List[PlanStep]
    complexity: str  # "simple", "moderate", "complex"
    estimated_duration_ms: int
    preconditions: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)


PLANNER_SYSTEM_PROMPT = """You are a task planner for a Mac desktop assistant. Given a goal and screen context, create an execution plan.

## YOUR TASK
Create a step-by-step plan with:
1. STEPS: Ordered actions to achieve the goal
2. VERIFICATION: How to check each step succeeded
3. FALLBACKS: Alternative approaches if steps fail
4. COMPLEXITY: simple (1 step), moderate (2-4), complex (5+)

## AVAILABLE ACTIONS
- click: {"action": "click", "x": X, "y": Y}
- double_click: {"action": "double_click", "x": X, "y": Y}
- type: {"action": "type", "text": "..."}
- press: {"action": "press", "key": "enter/escape/tab/etc"}
- hotkey: {"action": "hotkey", "keys": ["command", "c"]}
- scroll: {"action": "scroll", "amount": -300}  (negative=down)
- open_app: {"action": "open_app", "app": "Chrome"}
- open_url: {"action": "open_url", "url": "https://..."}
- wait: {"action": "wait", "seconds": 1}

## VERIFICATION RULES
- After click: "Element should be highlighted/selected"
- After open_app: "App window should be visible"
- After open_url: "Page should start loading"
- After type: "Text should appear in field"
- After scroll: "New content should be visible"

## FALLBACK STRATEGIES
- If click fails → try scrolling to find element, then click
- If app doesn't open → try Spotlight search
- If URL fails → try searching in Google
- If element not found → try alternative location

## OUTPUT FORMAT
{
  "goal": "...",
  "complexity": "simple|moderate|complex",
  "estimated_ms": 1000,
  "preconditions": ["Chrome must be installed"],
  "risks": ["May require login"],
  "steps": [
    {
      "action": {"action": "open_app", "app": "Chrome"},
      "description": "Open Chrome browser",
      "verification": "Chrome window visible",
      "fallback": {"action": "hotkey", "keys": ["command", "space"]}
    }
  ]
}

For SIMPLE tasks (scroll, click, open app), output just ONE step with done:true.

Respond with JSON only."""


class TaskPlanner:
    """Plans task execution with verification and fallbacks."""

    def __init__(self):
        self.client = get_anthropic_client(ANTHROPIC_API_KEY)

    def plan(
        self,
        intent: Intent,
        screen_description: Optional[str] = None,
        known_procedures: Optional[List[Dict]] = None
    ) -> Plan:
        """
        Create execution plan for an intent.

        Args:
            intent: Understood user intent
            screen_description: Description of current screen (if available)
            known_procedures: Similar procedures that worked before

        Returns:
            Plan with steps, verification, and fallbacks
        """
        # Check for simple intents first
        if intent.is_simple and intent.suggested_action:
            return self._simple_plan(intent)

        # Check if we have a known procedure for this
        if known_procedures:
            best_procedure = self._find_best_procedure(intent.goal, known_procedures)
            if best_procedure and best_procedure.get("success_rate", 0) > 0.7:
                return self._procedure_to_plan(best_procedure, intent.goal)

        # Generate new plan via Claude
        return self._generate_plan(intent, screen_description)

    def _simple_plan(self, intent: Intent) -> Plan:
        """Create a single-step plan for simple intents."""
        action = intent.suggested_action.copy()
        action["done"] = True

        return Plan(
            goal=intent.goal,
            steps=[
                PlanStep(
                    action=action,
                    description=intent.goal,
                    verification="Action completed",
                    fallback=None
                )
            ],
            complexity="simple",
            estimated_duration_ms=500
        )

    def _find_best_procedure(
        self,
        goal: str,
        procedures: List[Dict]
    ) -> Optional[Dict]:
        """Find the best matching procedure with highest success rate."""
        goal_lower = goal.lower()

        best = None
        best_score = 0

        for proc in procedures:
            trigger = proc.get("trigger", "").lower()
            # Simple matching - could use embeddings for better matching
            if trigger in goal_lower or goal_lower in trigger:
                success_rate = proc.get("success_rate", 0.5)
                if success_rate > best_score:
                    best = proc
                    best_score = success_rate

        return best

    def _procedure_to_plan(self, procedure: Dict, goal: str) -> Plan:
        """Convert a stored procedure to a Plan."""
        steps = []
        for step_data in procedure.get("steps", []):
            steps.append(PlanStep(
                action=step_data.get("action", {}),
                description=step_data.get("description", ""),
                verification=step_data.get("verification", "Check screen"),
                fallback=step_data.get("fallback")
            ))

        return Plan(
            goal=goal,
            steps=steps,
            complexity="moderate" if len(steps) <= 4 else "complex",
            estimated_duration_ms=procedure.get("avg_duration_ms", 2000),
            preconditions=procedure.get("preconditions", []),
            risks=procedure.get("risks", [])
        )

    def _generate_plan(
        self,
        intent: Intent,
        screen_description: Optional[str]
    ) -> Plan:
        """Generate a new plan using Claude."""
        prompt = f"""Goal: {intent.goal}
Confidence: {intent.confidence}
Resolved references: {intent.resolved_references}

{f"Current screen: {screen_description}" if screen_description else "Screen context not available yet."}

Create an execution plan. For simple tasks (scroll, click, open), use just ONE step.
Respond with JSON only."""

        try:
            response = self.client.messages.create(
                model=CLAUDE_MODEL_FAST,
                max_tokens=800,
                system=PLANNER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )

            return self._parse_plan(response.content[0].text, intent.goal)

        except Exception as e:
            print(f"Planning error: {e}")
            # Return minimal fallback plan
            return Plan(
                goal=intent.goal,
                steps=[],
                complexity="unknown",
                estimated_duration_ms=5000,
                risks=["Planning failed - will use reactive execution"]
            )

    def _parse_plan(self, response_text: str, goal: str) -> Plan:
        """Parse Claude's JSON response into Plan object."""
        try:
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            data = json.loads(text)

            steps = []
            for step_data in data.get("steps", []):
                steps.append(PlanStep(
                    action=step_data.get("action", {}),
                    description=step_data.get("description", ""),
                    verification=step_data.get("verification", ""),
                    fallback=step_data.get("fallback")
                ))

            return Plan(
                goal=data.get("goal", goal),
                steps=steps,
                complexity=data.get("complexity", "moderate"),
                estimated_duration_ms=data.get("estimated_ms", 2000),
                preconditions=data.get("preconditions", []),
                risks=data.get("risks", [])
            )

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse plan: {e}")
            return Plan(
                goal=goal,
                steps=[],
                complexity="unknown",
                estimated_duration_ms=5000
            )

    def adapt_plan(
        self,
        plan: Plan,
        failed_step_index: int,
        failure_reason: str
    ) -> Plan:
        """
        Adapt a plan after a step fails.

        Args:
            plan: Original plan
            failed_step_index: Which step failed
            failure_reason: Why it failed

        Returns:
            Modified plan with alternative approach
        """
        # If failed step has a fallback, use it
        failed_step = plan.steps[failed_step_index]
        if failed_step.fallback:
            new_steps = plan.steps.copy()
            new_steps[failed_step_index] = PlanStep(
                action=failed_step.fallback,
                description=f"Fallback: {failed_step.description}",
                verification=failed_step.verification,
                fallback=None  # No double fallback
            )
            return Plan(
                goal=plan.goal,
                steps=new_steps,
                complexity=plan.complexity,
                estimated_duration_ms=plan.estimated_duration_ms,
                preconditions=plan.preconditions,
                risks=plan.risks + [f"Original step {failed_step_index} failed: {failure_reason}"]
            )

        # No fallback - return plan with remaining steps
        return Plan(
            goal=plan.goal,
            steps=plan.steps[failed_step_index + 1:],
            complexity=plan.complexity,
            estimated_duration_ms=plan.estimated_duration_ms,
            preconditions=plan.preconditions,
            risks=plan.risks + [f"Step {failed_step_index} failed without fallback"]
        )


# Singleton
_planner: Optional[TaskPlanner] = None


def get_planner() -> TaskPlanner:
    """Get singleton TaskPlanner instance."""
    global _planner
    if _planner is None:
        _planner = TaskPlanner()
    return _planner
