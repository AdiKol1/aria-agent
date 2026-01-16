"""
Coordinator agent that routes requests to specialists.
"""

from typing import Dict, Optional, List
from .base import BaseAgent, AgentContext, AgentResult

class Coordinator:
    """Routes requests to specialist agents using Swarm-style handoffs."""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.default_agent = "system"
        self.max_handoffs = 5

    def register_agent(self, agent: BaseAgent):
        """Register a specialist agent."""
        self.agents[agent.name] = agent
        print(f"Registered agent: {agent.name}")

    def route(self, context: AgentContext) -> str:
        """Determine which agent should handle the request."""
        scores = {}

        for name, agent in self.agents.items():
            scores[name] = agent.matches(context.user_input)

        # Get highest scoring agent
        if scores:
            best_agent = max(scores, key=scores.get)
            if scores[best_agent] > 0.5:
                return best_agent

        return self.default_agent

    async def process(self, context: AgentContext) -> AgentResult:
        """Process request with automatic handoffs."""
        current_agent_name = self.route(context)
        handoff_count = 0

        while handoff_count < self.max_handoffs:
            if current_agent_name not in self.agents:
                return AgentResult.error(f"Unknown agent: {current_agent_name}")

            agent = self.agents[current_agent_name]
            print(f"[Coordinator] Routing to {agent.name}: {context.user_input[:50]}...")

            try:
                result = await agent.process(context)
            except Exception as e:
                return AgentResult.error(f"Agent {agent.name} error: {e}")

            # Check for handoff
            if result.handoff_to:
                if result.handoff_to not in self.agents:
                    return AgentResult.error(f"Invalid handoff target: {result.handoff_to}")

                # Record history
                context.history.append({
                    "agent": agent.name,
                    "response": result.response,
                    "handoff_to": result.handoff_to
                })

                current_agent_name = result.handoff_to
                handoff_count += 1
                print(f"[Coordinator] Handoff to {result.handoff_to}")
            else:
                return result

        return AgentResult.error("Too many handoffs")

    def list_agents(self) -> List[Dict]:
        """List all registered agents."""
        return [
            {
                "name": agent.name,
                "description": agent.description,
                "triggers": agent.triggers
            }
            for agent in self.agents.values()
        ]


# Singleton
_coordinator: Optional[Coordinator] = None

def get_coordinator() -> Coordinator:
    """Get the global coordinator."""
    global _coordinator
    if _coordinator is None:
        _coordinator = Coordinator()

        # Register all agents
        from .file_agent import FileAgent
        from .browser_agent import BrowserAgent
        from .system_agent import SystemAgent
        from .code_agent import CodeAgent

        _coordinator.register_agent(FileAgent())
        _coordinator.register_agent(BrowserAgent())
        _coordinator.register_agent(SystemAgent())
        _coordinator.register_agent(CodeAgent())

    return _coordinator
