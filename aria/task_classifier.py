"""
Task Classifier for Smart Model Selection

Classifies tasks as simple or complex to use the appropriate Claude model:
- Simple tasks: Use Sonnet (2-4s latency)
- Complex tasks: Use Opus (4-8s latency)
"""

from typing import Tuple
import re

# Simple task patterns - these can use the faster Sonnet model
SIMPLE_PATTERNS = [
    r"^open\s+\w+",  # "open chrome", "open safari"
    r"^go\s+to\s+\w+",  # "go to youtube"
    r"^click\s+",  # "click the button"
    r"^type\s+",  # "type hello"
    r"^scroll\s+",  # "scroll down"
    r"^close\s+",  # "close the window"
    r"^minimize\s+",  # "minimize"
    r"^maximize\s+",  # "maximize"
    r"^switch\s+to\s+",  # "switch to chrome"
    r"^search\s+for\s+\w+$",  # Simple search (not multi-step)
    r"^press\s+",  # "press enter"
    r"^navigate\s+to\s+",  # "navigate to google.com"
]

# Complex task patterns - these need Opus for reasoning
COMPLEX_PATTERNS = [
    r"and\s+then\s+",  # Multi-step tasks
    r"find\s+.*\s+and\s+",  # Find and do something
    r"compare\s+",  # Comparison tasks
    r"analyze\s+",  # Analysis tasks
    r"debug\s+",  # Debugging
    r"fix\s+",  # Fixing issues
    r"figure\s+out",  # Reasoning required
    r"what\s+is\s+the\s+best",  # Decision making
    r"how\s+do\s+i",  # Instructional (might need reasoning)
    r"tell\s+me\s+about",  # Information retrieval with reasoning
]


def classify_task(task: str) -> Tuple[str, float]:
    """
    Classify a task as simple or complex.

    Args:
        task: The task description

    Returns:
        Tuple of (classification, confidence)
        classification: "simple" or "complex"
        confidence: 0.0 to 1.0
    """
    task_lower = task.lower().strip()

    # Check for complex patterns first (they take priority)
    for pattern in COMPLEX_PATTERNS:
        if re.search(pattern, task_lower):
            return ("complex", 0.8)

    # Check for simple patterns
    for pattern in SIMPLE_PATTERNS:
        if re.search(pattern, task_lower):
            return ("simple", 0.9)

    # Word count heuristic - longer tasks are usually more complex
    word_count = len(task_lower.split())
    if word_count <= 5:
        return ("simple", 0.6)
    elif word_count <= 10:
        return ("simple", 0.5)
    else:
        return ("complex", 0.6)


def get_model_for_task(task: str, simple_model: str, complex_model: str) -> str:
    """
    Get the appropriate model for a task.

    Args:
        task: The task description
        simple_model: Model to use for simple tasks
        complex_model: Model to use for complex tasks

    Returns:
        The model name to use
    """
    classification, confidence = classify_task(task)

    # Only use simple model if we're confident it's simple
    if classification == "simple" and confidence >= 0.7:
        return simple_model

    return complex_model


# Test examples
if __name__ == "__main__":
    test_tasks = [
        "open chrome",
        "go to youtube",
        "search for anthropic",
        "open chrome, go to google, search for AI, and tell me the first result",
        "find the error in this code and fix it",
        "click the submit button",
        "what is the best way to optimize this?",
    ]

    for task in test_tasks:
        classification, confidence = classify_task(task)
        print(f"{task[:50]:50} -> {classification} ({confidence:.1f})")
