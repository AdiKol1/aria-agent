"""
Memory Skills for Aria.

Provides memory operations:
- Remember facts and preferences
- Recall information
- List all memories
- Search memory by topic
"""

import re
from aria.skills.registry import skill
from aria.skills.base import SkillCategory, SkillContext, SkillResult


@skill(
    name="remember_fact",
    description="Remember a fact or preference for later recall",
    triggers=["remember", "note", "save", "store"],
    category=SkillCategory.MEMORY,
)
def remember_fact(context: SkillContext) -> SkillResult:
    """Store a fact in memory."""
    from aria.memory import get_memory

    memory = get_memory()
    input_text = context.user_input

    # Extract what to remember
    patterns = [
        r"remember\s+(?:that\s+)?(.+)",
        r"note\s+(?:that\s+)?(.+)",
        r"save\s+(?:that\s+)?(.+)",
        r"store\s+(?:that\s+)?(.+)",
    ]

    fact = None
    for pattern in patterns:
        match = re.search(pattern, input_text, re.IGNORECASE)
        if match:
            fact = match.group(1).strip()
            break

    if not fact:
        return SkillResult.fail("Couldn't determine what to remember")

    # Determine category
    category = "other"
    if any(word in input_text.lower() for word in ["prefer", "like", "favorite", "always", "never"]):
        category = "preference"
    elif any(word in input_text.lower() for word in ["name is", "i am", "my", "i'm"]):
        category = "personal"
    elif any(word in input_text.lower() for word in ["work", "job", "project", "task"]):
        category = "work"

    success = memory.remember_fact(fact, category=category)
    if success:
        return SkillResult.ok(f"I'll remember that: {fact}")
    else:
        return SkillResult.fail("Failed to save to memory")


@skill(
    name="recall_memory",
    description="Recall information from memory based on a query",
    triggers=["recall", "what do you know", "what did I say", "remind me"],
    category=SkillCategory.MEMORY,
)
def recall_memory(context: SkillContext) -> SkillResult:
    """Recall information from memory."""
    from aria.memory import get_memory

    memory = get_memory()
    input_text = context.user_input

    # Extract query
    patterns = [
        r"recall\s+(?:about\s+)?(.+)",
        r"what do you know about\s+(.+)",
        r"what did I say about\s+(.+)",
        r"remind me about\s+(.+)",
        r"what do you remember about\s+(.+)",
    ]

    query = None
    for pattern in patterns:
        match = re.search(pattern, input_text, re.IGNORECASE)
        if match:
            query = match.group(1).strip()
            break

    if not query:
        # Default to using the whole input as query
        query = input_text

    facts = memory.recall_facts(query, n_results=5)

    if not facts:
        return SkillResult.ok("I don't have any relevant memories about that.")

    # Format facts for output
    lines = ["Here's what I remember:"]
    for f in facts:
        confidence = f.get('confidence', 0.5)
        relevance = f.get('relevance', 0)
        if confidence > 0.5 and relevance > 0.3:
            lines.append(f"- {f['fact']}")

    if len(lines) == 1:
        return SkillResult.ok("I don't have any confident memories about that.")

    return SkillResult.ok(
        "\n".join(lines),
        data={"facts": facts}
    )


@skill(
    name="list_all_memories",
    description="List everything stored in memory",
    triggers=["list memories", "all memories", "everything you know", "what do you know"],
    category=SkillCategory.MEMORY,
)
def list_all_memories(context: SkillContext) -> SkillResult:
    """List all stored facts."""
    from aria.memory import get_memory

    memory = get_memory()
    facts = memory.get_all_facts()

    if not facts:
        return SkillResult.ok("I don't have any memories stored yet.")

    lines = [f"I have {len(facts)} memories:"]
    for i, fact in enumerate(facts[:20], 1):  # Limit to first 20
        lines.append(f"{i}. {fact}")

    if len(facts) > 20:
        lines.append(f"... and {len(facts) - 20} more")

    return SkillResult.ok(
        "\n".join(lines),
        data={"facts": facts, "count": len(facts)}
    )


@skill(
    name="forget_memory",
    description="Remove a specific memory (requires confirmation)",
    triggers=["forget", "delete memory", "remove memory"],
    category=SkillCategory.MEMORY,
    requires_confirmation=True,
    is_destructive=True,
)
def forget_memory(context: SkillContext) -> SkillResult:
    """Remove a memory (with confirmation)."""
    from aria.memory import get_memory

    memory = get_memory()
    input_text = context.user_input

    # Extract what to forget
    patterns = [
        r"forget\s+(?:that\s+)?(.+)",
        r"delete\s+(?:memory\s+)?(?:about\s+)?(.+)",
        r"remove\s+(?:memory\s+)?(?:about\s+)?(.+)",
    ]

    query = None
    for pattern in patterns:
        match = re.search(pattern, input_text, re.IGNORECASE)
        if match:
            query = match.group(1).strip()
            break

    if not query:
        return SkillResult.fail("Couldn't determine what to forget")

    # Find matching facts
    facts = memory.recall_facts(query, n_results=1)

    if not facts:
        return SkillResult.ok(f"I don't have any memories matching '{query}'")

    # For now, just confirm - actual deletion would need memory.delete_fact()
    fact = facts[0]['fact']
    return SkillResult.confirm(
        f"Do you want me to forget: \"{fact}\"?"
    )


@skill(
    name="clear_all_memories",
    description="Clear all memories (requires confirmation)",
    triggers=["clear all memories", "forget everything", "reset memory"],
    category=SkillCategory.MEMORY,
    requires_confirmation=True,
    is_destructive=True,
)
def clear_all_memories(context: SkillContext) -> SkillResult:
    """Clear all memories (with confirmation)."""
    return SkillResult.confirm(
        "Are you sure you want to clear ALL memories? This cannot be undone."
    )


@skill(
    name="search_memory",
    description="Search memory for specific topics or keywords",
    triggers=["search memory", "find in memory", "look up"],
    category=SkillCategory.MEMORY,
)
def search_memory(context: SkillContext) -> SkillResult:
    """Search memory by topic."""
    from aria.memory import get_memory

    memory = get_memory()
    input_text = context.user_input

    # Extract search query
    patterns = [
        r"search\s+(?:memory\s+)?(?:for\s+)?(.+)",
        r"find\s+(?:in memory\s+)?(.+)",
        r"look up\s+(.+)",
    ]

    query = None
    for pattern in patterns:
        match = re.search(pattern, input_text, re.IGNORECASE)
        if match:
            query = match.group(1).strip()
            break

    if not query:
        query = input_text

    facts = memory.recall_facts(query, n_results=10)

    if not facts:
        return SkillResult.ok(f"No memories found for '{query}'")

    lines = [f"Found {len(facts)} related memories:"]
    for f in facts:
        relevance = f.get('relevance', 0)
        confidence = f.get('confidence', 0.5)
        lines.append(f"- {f['fact']} (relevance: {relevance:.0%}, confidence: {confidence:.0%})")

    return SkillResult.ok(
        "\n".join(lines),
        data={"facts": facts, "query": query}
    )
