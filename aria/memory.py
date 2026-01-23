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

from .config import ANTHROPIC_API_KEY, DATA_PATH
from .lazy_anthropic import get_client as get_anthropic_client


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
        self.claude = get_anthropic_client(ANTHROPIC_API_KEY)

        print(f"Memory initialized: {self.facts.count()} facts, {self.interactions.count()} interactions")

    def _generate_id(self, text: str) -> str:
        """Generate a unique ID for a memory."""
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def remember_fact(
        self,
        fact: str,
        category: str = "general",
        confidence: float = 0.8
    ) -> bool:
        """
        Store a fact about the user with enhanced metadata.

        Args:
            fact: The fact to remember (e.g., "User's name is John")
            category: Category like "preference", "personal", "work", etc.
            confidence: How confident we are about this fact (0.0 to 1.0)
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
                # Get existing metadata to preserve usage count
                old_meta = existing['metadatas'][0][0] if existing['metadatas'] else {}
                usage_count = old_meta.get('usage_count', 0)
                old_confidence = old_meta.get('confidence', 0.5)

                # Increase confidence if fact is being reinforced
                new_confidence = min(1.0, (old_confidence + confidence) / 2 + 0.1)

                # Update existing
                self.facts.update(
                    ids=[existing['ids'][0][0]],
                    documents=[fact],
                    metadatas=[{
                        "category": category,
                        "confidence": new_confidence,
                        "usage_count": usage_count,
                        "updated_at": datetime.now().isoformat(),
                        "created_at": old_meta.get('created_at', datetime.now().isoformat()),
                    }]
                )
                print(f"Updated fact (confidence: {new_confidence:.2f}): {fact[:50]}...")
            else:
                # Add new fact
                self.facts.add(
                    ids=[fact_id],
                    documents=[fact],
                    metadatas=[{
                        "category": category,
                        "confidence": confidence,
                        "usage_count": 0,
                        "success_rate": 1.0,  # Track when using this fact leads to success
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                    }]
                )
                print(f"Remembered fact (confidence: {confidence:.2f}): {fact[:50]}...")

            return True
        except Exception as e:
            print(f"Error remembering fact: {e}")
            return False

    def mark_fact_used(self, fact_id: str, task_succeeded: bool = True) -> None:
        """
        Mark a fact as having been used, updating usage stats.

        Args:
            fact_id: The ID of the fact that was used
            task_succeeded: Whether the task using this fact succeeded
        """
        try:
            # Get current metadata
            result = self.facts.get(ids=[fact_id])
            if not result['documents']:
                return

            meta = result['metadatas'][0] if result['metadatas'] else {}
            usage_count = meta.get('usage_count', 0) + 1
            old_success_rate = meta.get('success_rate', 1.0)

            # Update success rate (rolling average)
            new_success_rate = (old_success_rate * (usage_count - 1) + (1 if task_succeeded else 0)) / usage_count

            # Update confidence based on success rate
            confidence = meta.get('confidence', 0.5)
            if task_succeeded:
                confidence = min(1.0, confidence + 0.02)
            else:
                confidence = max(0.1, confidence - 0.05)

            self.facts.update(
                ids=[fact_id],
                metadatas=[{
                    **meta,
                    "usage_count": usage_count,
                    "success_rate": new_success_rate,
                    "confidence": confidence,
                    "last_used": datetime.now().isoformat(),
                }]
            )
        except Exception as e:
            print(f"Error marking fact used: {e}")

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
                meta = results['metadatas'][0][i]
                facts.append({
                    "id": results['ids'][0][i],
                    "fact": doc,
                    "category": meta.get('category', 'general'),
                    "confidence": meta.get('confidence', 0.5),
                    "usage_count": meta.get('usage_count', 0),
                    "success_rate": meta.get('success_rate', 1.0),
                    "relevance": 1 - results['distances'][0][i] if results['distances'] else 0
                })

            # Sort by combined score: relevance * confidence * success_rate
            for f in facts:
                f['score'] = f['relevance'] * f['confidence'] * f['success_rate']
            facts.sort(key=lambda x: x['score'], reverse=True)

            return facts
        except Exception as e:
            print(f"Error recalling facts: {e}")
            return []

    def search_memories(self, query: str, n_results: int = 5) -> List[str]:
        """
        Search for relevant memory facts and return as simple list of strings.
        Used by IntentEngine for context resolution.

        Args:
            query: What to search for
            n_results: Maximum number of results

        Returns:
            List of fact strings, ordered by relevance and quality
        """
        facts = self.recall_facts(query, n_results)
        # Return only high-quality facts (confidence > 0.3 and relevance > 0.3)
        return [
            f['fact'] for f in facts
            if f.get('confidence', 0.5) > 0.3 and f.get('relevance', 0) > 0.3
        ]

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

    def get_startup_context(self, max_facts: int = 15, max_procedures: int = 5) -> str:
        """
        Get general context to inject at conversation start.

        This provides Aria with relevant knowledge about the user
        BEFORE the conversation begins, enabling her to be more
        personalized and effective from the first interaction.

        Args:
            max_facts: Maximum number of facts to include
            max_procedures: Maximum number of procedures to include

        Returns:
            Formatted context string with user knowledge
        """
        context_parts = []

        try:
            # CRITICAL: First, search for identity/name facts specifically
            identity_facts = []
            try:
                name_results = self.facts.query(
                    query_texts=["user's name is called prefer to be called"],
                    n_results=10
                )
                if name_results and name_results['documents'] and name_results['documents'][0]:
                    for doc in name_results['documents'][0]:
                        doc_lower = doc.lower()
                        # Only include actual identity facts, not action outcomes
                        if 'action_outcome' not in doc_lower and any(kw in doc_lower for kw in ['name is', 'called', 'prefer']):
                            identity_facts.append(f"- {doc}")
            except Exception as e:
                print(f"Error searching identity facts: {e}")

            # Get user preferences and facts
            all_facts = self.get_all_facts()
            if all_facts:
                # Prioritize preferences and personal info, but filter out action_outcomes
                preference_facts = []
                personal_facts = []
                other_facts = []

                for fact in all_facts[:max_facts * 3]:  # Get more, then filter
                    fact_lower = fact.lower()
                    # Skip action_outcome entries - they clutter the context
                    if fact_lower.startswith('action ') or 'succeeded with' in fact_lower or 'failed with' in fact_lower:
                        continue
                    if any(word in fact_lower for word in ['prefer', 'like', 'favorite', 'always', 'usually', 'wants']):
                        if f"- {fact}" not in identity_facts:  # Avoid duplicates
                            preference_facts.append(f"- {fact}")
                    elif any(word in fact_lower for word in ['name', 'called', 'personal', 'knows', 'has dog', 'has cat']):
                        if f"- {fact}" not in identity_facts:  # Avoid duplicates
                            personal_facts.append(f"- {fact}")
                    else:
                        other_facts.append(f"- {fact}")

                # Identity first, then preferences, then personal, then others
                selected = identity_facts[:3] + preference_facts[:5] + personal_facts[:4] + other_facts[:3]
                if selected:
                    context_parts.append("## What I Know About You\n" + "\n".join(selected[:max_facts]))

            # Get learned procedures (things I know how to do)
            if self.procedures.count() > 0:
                results = self.procedures.get(limit=max_procedures)
                if results and results['documents']:
                    procedures = [f"- {doc}" for doc in results['documents']]
                    context_parts.append("## Procedures I've Learned\n" + "\n".join(procedures))

            # Get recent interaction insights (pattern recognition)
            if self.interactions.count() > 0:
                results = self.interactions.get(limit=5)
                if results and results['metadatas']:
                    # Look for successful patterns
                    successes = []
                    for i, meta in enumerate(results['metadatas']):
                        if meta.get('outcome') == 'success':
                            summary = results['documents'][i] if results['documents'] else ""
                            if summary:
                                successes.append(f"- {summary}")
                    if successes:
                        context_parts.append("## Recent Successful Interactions\n" + "\n".join(successes[:3]))

        except Exception as e:
            print(f"Error getting startup context: {e}")

        if context_parts:
            return "\n\n".join(context_parts)
        return ""

    def record_action_outcome(self, action_name: str, action_args: dict, success: bool, context: str = "") -> None:
        """
        Record the outcome of an action for learning what works.

        This builds a database of what actions succeed/fail in what contexts,
        enabling Aria to choose better approaches over time.

        Args:
            action_name: The action/tool that was used (e.g., "hotkey", "click")
            action_args: The arguments passed to the action
            success: Whether the action succeeded
            context: Additional context (e.g., active app, user goal)
        """
        try:
            # Create action signature for pattern matching
            args_summary = json.dumps(action_args, sort_keys=True)[:200]  # Truncate long args
            action_signature = f"{action_name}:{args_summary}"

            # Store as a fact with special format for action tracking
            outcome = "succeeded" if success else "failed"
            fact = f"Action {action_name} {outcome}"
            if action_args:
                fact += f" with {args_summary}"
            if context:
                fact += f" in context: {context}"

            # Use category "action_outcome" for easy retrieval
            self.remember_fact(fact, "action_outcome")

            print(f"[Learning] Recorded action outcome: {action_name} -> {outcome}")

        except Exception as e:
            print(f"Error recording action outcome: {e}")

    def get_action_success_rate(self, action_name: str, similar_args: dict = None) -> Optional[dict]:
        """
        Get the historical success rate for an action type.

        Args:
            action_name: The action to check
            similar_args: Optional args to match similar past actions

        Returns:
            Dict with success_rate, total_count, and recent_outcomes
        """
        try:
            # Search for this action type in outcomes
            query = f"Action {action_name}"
            results = self.facts.query(
                query_texts=[query],
                n_results=20,
                where={"category": "action_outcome"} if self.facts.count() > 0 else None
            )

            if not results or not results['documents'] or not results['documents'][0]:
                return None

            outcomes = results['documents'][0]
            successes = sum(1 for o in outcomes if "succeeded" in o.lower())
            failures = sum(1 for o in outcomes if "failed" in o.lower())
            total = successes + failures

            if total == 0:
                return None

            return {
                "action": action_name,
                "success_rate": successes / total,
                "successes": successes,
                "failures": failures,
                "total": total,
                "recent_outcomes": outcomes[:5]
            }

        except Exception as e:
            print(f"Error getting action success rate: {e}")
            return None

    def get_best_approach(self, goal: str) -> Optional[str]:
        """
        Suggest the best approach for a goal based on past success rates.

        Args:
            goal: What the user wants to accomplish

        Returns:
            Suggestion string with recommended approach, or None
        """
        try:
            # Search for relevant past successes
            results = self.facts.query(
                query_texts=[goal],
                n_results=10,
                where={"category": "action_outcome"} if self.facts.count() > 0 else None
            )

            if not results or not results['documents'] or not results['documents'][0]:
                return None

            # Filter for successes
            successful_approaches = [
                doc for doc in results['documents'][0]
                if "succeeded" in doc.lower()
            ]

            if successful_approaches:
                return f"Based on past experience, these approaches have worked: {'; '.join(successful_approaches[:3])}"

            return None

        except Exception as e:
            print(f"Error getting best approach: {e}")
            return None

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
