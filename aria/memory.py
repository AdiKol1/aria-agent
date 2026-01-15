"""
Aria Memory System

Long-term memory using ChromaDB for semantic search.
Remembers facts, preferences, and past interactions.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import chromadb
from chromadb.config import Settings
import anthropic

from .config import ANTHROPIC_API_KEY, DATA_PATH


# Memory storage path
MEMORY_PATH = DATA_PATH / "memory"
MEMORY_PATH.mkdir(parents=True, exist_ok=True)


class AriaMemory:
    """Long-term memory for Aria using ChromaDB."""

    def __init__(self):
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(MEMORY_PATH / "chromadb"),
            settings=Settings(anonymized_telemetry=False)
        )

        # Collections for different memory types
        self.facts = self.client.get_or_create_collection(
            name="facts",
            metadata={"description": "Facts about the user and their preferences"}
        )

        self.interactions = self.client.get_or_create_collection(
            name="interactions",
            metadata={"description": "Past conversation summaries"}
        )

        self.procedures = self.client.get_or_create_collection(
            name="procedures",
            metadata={"description": "Learned workflows and patterns"}
        )

        # Claude client for memory extraction
        self.claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        print(f"Memory initialized: {self.facts.count()} facts, {self.interactions.count()} interactions")

    def _generate_id(self, text: str) -> str:
        """Generate a unique ID for a memory."""
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def remember_fact(self, fact: str, category: str = "general") -> bool:
        """
        Store a fact about the user.

        Args:
            fact: The fact to remember (e.g., "User's name is John")
            category: Category like "preference", "personal", "work", etc.
        """
        try:
            fact_id = self._generate_id(fact)

            # Check if similar fact exists
            existing = self.facts.query(
                query_texts=[fact],
                n_results=1
            )

            # If very similar fact exists (distance < 0.1), update it
            if existing['distances'] and existing['distances'][0] and existing['distances'][0][0] < 0.1:
                # Update existing
                self.facts.update(
                    ids=[existing['ids'][0][0]],
                    documents=[fact],
                    metadatas=[{
                        "category": category,
                        "updated_at": datetime.now().isoformat(),
                    }]
                )
                print(f"Updated fact: {fact[:50]}...")
            else:
                # Add new fact
                self.facts.add(
                    ids=[fact_id],
                    documents=[fact],
                    metadatas=[{
                        "category": category,
                        "created_at": datetime.now().isoformat(),
                    }]
                )
                print(f"Remembered fact: {fact[:50]}...")

            return True
        except Exception as e:
            print(f"Error remembering fact: {e}")
            return False

    def remember_interaction(self, summary: str, user_request: str, outcome: str) -> bool:
        """
        Store a summary of an interaction.

        Args:
            summary: Brief summary of what happened
            user_request: What the user asked for
            outcome: Whether it succeeded and what was done
        """
        try:
            interaction_id = self._generate_id(f"{datetime.now().isoformat()}-{summary}")

            self.interactions.add(
                ids=[interaction_id],
                documents=[summary],
                metadatas=[{
                    "user_request": user_request[:200],
                    "outcome": outcome[:200],
                    "timestamp": datetime.now().isoformat(),
                }]
            )
            print(f"Remembered interaction: {summary[:50]}...")
            return True
        except Exception as e:
            print(f"Error remembering interaction: {e}")
            return False

    def remember_procedure(self, trigger: str, steps: List[str], description: str) -> bool:
        """
        Store a learned procedure/workflow.

        Args:
            trigger: What triggers this procedure (e.g., "check email")
            steps: List of steps to execute
            description: Human-readable description
        """
        try:
            proc_id = self._generate_id(trigger)

            self.procedures.upsert(
                ids=[proc_id],
                documents=[description],
                metadatas=[{
                    "trigger": trigger,
                    "steps": json.dumps(steps),
                    "updated_at": datetime.now().isoformat(),
                }]
            )
            print(f"Remembered procedure: {trigger}")
            return True
        except Exception as e:
            print(f"Error remembering procedure: {e}")
            return False

    def recall_facts(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Recall relevant facts based on a query.

        Args:
            query: What to search for
            n_results: Maximum number of facts to return

        Returns:
            List of relevant facts with metadata
        """
        try:
            if self.facts.count() == 0:
                return []

            results = self.facts.query(
                query_texts=[query],
                n_results=min(n_results, self.facts.count())
            )

            facts = []
            for i, doc in enumerate(results['documents'][0]):
                facts.append({
                    "fact": doc,
                    "category": results['metadatas'][0][i].get('category', 'general'),
                    "relevance": 1 - results['distances'][0][i] if results['distances'] else 0
                })

            return facts
        except Exception as e:
            print(f"Error recalling facts: {e}")
            return []

    def recall_interactions(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Recall relevant past interactions.

        Args:
            query: What to search for
            n_results: Maximum number of interactions to return

        Returns:
            List of relevant interactions with metadata
        """
        try:
            if self.interactions.count() == 0:
                return []

            results = self.interactions.query(
                query_texts=[query],
                n_results=min(n_results, self.interactions.count())
            )

            interactions = []
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                interactions.append({
                    "summary": doc,
                    "user_request": meta.get('user_request', ''),
                    "outcome": meta.get('outcome', ''),
                    "timestamp": meta.get('timestamp', ''),
                })

            return interactions
        except Exception as e:
            print(f"Error recalling interactions: {e}")
            return []

    def recall_procedure(self, trigger: str) -> Optional[Dict[str, Any]]:
        """
        Recall a procedure by its trigger.

        Args:
            trigger: The trigger phrase

        Returns:
            Procedure details or None
        """
        try:
            if self.procedures.count() == 0:
                return None

            results = self.procedures.query(
                query_texts=[trigger],
                n_results=1
            )

            if results['documents'] and results['documents'][0]:
                meta = results['metadatas'][0][0]
                return {
                    "description": results['documents'][0][0],
                    "trigger": meta.get('trigger', ''),
                    "steps": json.loads(meta.get('steps', '[]')),
                }
            return None
        except Exception as e:
            print(f"Error recalling procedure: {e}")
            return None

    def get_context_for_request(self, user_input: str) -> str:
        """
        Get relevant memory context for a user request.

        Args:
            user_input: What the user is asking

        Returns:
            Formatted memory context string
        """
        context_parts = []

        # Recall relevant facts
        facts = self.recall_facts(user_input, n_results=5)
        if facts:
            fact_list = [f"- {f['fact']}" for f in facts if f['relevance'] > 0.3]
            if fact_list:
                context_parts.append("Things I remember about you:\n" + "\n".join(fact_list))

        # Recall relevant past interactions
        interactions = self.recall_interactions(user_input, n_results=2)
        if interactions:
            interaction_list = [f"- {i['summary']}" for i in interactions]
            if interaction_list:
                context_parts.append("Relevant past interactions:\n" + "\n".join(interaction_list))

        # Check for matching procedure
        procedure = self.recall_procedure(user_input)
        if procedure:
            context_parts.append(f"I know a procedure for this: {procedure['description']}")

        return "\n\n".join(context_parts) if context_parts else ""

    def extract_and_store_memories(self, user_input: str, assistant_response: str, actions_taken: List[dict]) -> None:
        """
        Extract important information from a conversation turn and store it.

        Args:
            user_input: What the user said
            assistant_response: What Aria said
            actions_taken: List of actions executed
        """
        try:
            # Use Claude to extract memories
            extraction_prompt = f"""Analyze this conversation turn and extract any important information to remember.

User said: "{user_input}"
Assistant responded: "{assistant_response}"
Actions taken: {json.dumps(actions_taken) if actions_taken else "None"}

Extract:
1. FACTS: Any facts about the user (name, preferences, work, habits, etc.)
2. INTERACTION_SUMMARY: A brief summary of what happened (if significant)
3. PROCEDURE: If the user taught a new workflow or shortcut

Respond in JSON format:
{{
    "facts": [
        {{"fact": "...", "category": "preference|personal|work|habit|other"}}
    ],
    "interaction_summary": "..." or null if trivial,
    "procedure": {{"trigger": "...", "description": "...", "steps": [...]}} or null
}}

Only include meaningful, reusable information. Skip trivial interactions like "what's on screen" or simple greetings.
Respond with ONLY the JSON, no other text."""

            response = self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": extraction_prompt}]
            )

            # Parse the response
            try:
                extracted = json.loads(response.content[0].text)
            except json.JSONDecodeError:
                # Try to find JSON in response
                text = response.content[0].text
                start = text.find('{')
                end = text.rfind('}') + 1
                if start >= 0 and end > start:
                    extracted = json.loads(text[start:end])
                else:
                    return

            # Store extracted facts
            for fact_data in extracted.get('facts', []):
                if fact_data.get('fact'):
                    self.remember_fact(
                        fact_data['fact'],
                        fact_data.get('category', 'general')
                    )

            # Store interaction summary
            summary = extracted.get('interaction_summary')
            if summary:
                outcome = "success" if actions_taken else "conversation only"
                self.remember_interaction(summary, user_input, outcome)

            # Store procedure
            procedure = extracted.get('procedure')
            if procedure and procedure.get('trigger'):
                self.remember_procedure(
                    procedure['trigger'],
                    procedure.get('steps', []),
                    procedure.get('description', '')
                )

        except Exception as e:
            print(f"Error extracting memories: {e}")

    def get_all_facts(self) -> List[str]:
        """Get all stored facts."""
        try:
            if self.facts.count() == 0:
                return []
            results = self.facts.get()
            return results['documents'] if results['documents'] else []
        except Exception as e:
            print(f"Error getting facts: {e}")
            return []

    def clear_all(self) -> bool:
        """Clear all memories (use with caution!)."""
        try:
            self.client.delete_collection("facts")
            self.client.delete_collection("interactions")
            self.client.delete_collection("procedures")

            # Recreate collections
            self.facts = self.client.create_collection(name="facts")
            self.interactions = self.client.create_collection(name="interactions")
            self.procedures = self.client.create_collection(name="procedures")

            print("All memories cleared")
            return True
        except Exception as e:
            print(f"Error clearing memories: {e}")
            return False


# Singleton
_memory: Optional[AriaMemory] = None


def get_memory() -> AriaMemory:
    """Get the singleton AriaMemory instance."""
    global _memory
    if _memory is None:
        _memory = AriaMemory()
    return _memory
