"""
Browser and web operations specialist agent.
"""

from .base import BaseAgent, AgentContext, AgentResult
import webbrowser

class BrowserAgent(BaseAgent):
    """Handles web operations: search, navigate, browse."""

    name = "browser"
    description = "Handles web tasks like searching, navigating to URLs, and browsing"
    triggers = [
        "search", "google", "look up", "find online",
        "website", "web page", "url", "browse",
        "open site", "go to", "navigate",
        "download", "fetch"
    ]

    async def process(self, context: AgentContext) -> AgentResult:
        """Process browser/web request."""
        input_lower = context.user_input.lower()

        if any(w in input_lower for w in ["search", "google", "look up"]):
            return await self._web_search(context)
        elif any(w in input_lower for w in ["go to", "open", "navigate", "browse"]):
            return await self._open_url(context)
        else:
            # Default to search
            return await self._web_search(context)

    async def _web_search(self, context: AgentContext) -> AgentResult:
        """Perform a web search."""
        # Extract search query
        query = context.user_input

        # Remove common prefixes
        for prefix in ["search for", "google", "look up", "search"]:
            if query.lower().startswith(prefix):
                query = query[len(prefix):].strip()
                break

        if not query:
            return AgentResult.error("What should I search for?")

        # Open Google search
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"

        try:
            webbrowser.open(search_url)
            return AgentResult.ok(f"Searching Google for: {query}")
        except Exception as e:
            return AgentResult.error(f"Failed to open browser: {e}")

    async def _open_url(self, context: AgentContext) -> AgentResult:
        """Open a URL in the browser."""
        import re

        # Extract URL
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        match = re.search(url_pattern, context.user_input)

        if match:
            url = match.group(0)
        else:
            # Try to construct URL from domain-like text
            words = context.user_input.split()
            for word in words:
                if "." in word and not word.startswith("."):
                    url = f"https://{word}" if not word.startswith("http") else word
                    break
            else:
                return AgentResult.error("I couldn't find a URL to open.")

        try:
            webbrowser.open(url)
            return AgentResult.ok(f"Opening {url}")
        except Exception as e:
            return AgentResult.error(f"Failed to open URL: {e}")
