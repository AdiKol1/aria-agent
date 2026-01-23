"""
Aria Ambient Intelligence - Cortex

The Cortex is the synthesis layer that processes signals from watchers,
scores their relevance to the user's worlds, and generates insights.

Components:
- PriorityCalculator: Calculates priority of insights
- InsightGenerator: Generates actionable insights from signals
- ConnectionDetector: Finds opportunities across different worlds
"""

from .priority import PriorityCalculator
from .synthesis import InsightGenerator, ConnectionDetector

__all__ = [
    "PriorityCalculator",
    "InsightGenerator",
    "ConnectionDetector",
]
