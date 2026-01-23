"""
Aria Learning System

Advanced learning capabilities:
- Skill Learning: Record demonstrations, replay with adaptation
- Pattern Learning: Learn from corrections and repeated behaviors
- Memory Pruning: Intelligent lifecycle management for memories

Based on aria-memory TypeScript implementation, ported to Python.
"""

from .types import (
    ActionType,
    RecordedAction,
    LearnedSkill,
    RecordingSession,
    ExecutionContext,
    ExecutionResult,
    ObservationType,
    Observation,
    LearnedPattern,
    MemoryType,
    PruningPolicy,
)
from .skill_recorder import SkillRecorder
from .skill_executor import SkillExecutor
from .patterns import PatternLearner
from .pruner import MemoryPruner

__all__ = [
    # Types
    "ActionType",
    "RecordedAction",
    "LearnedSkill",
    "RecordingSession",
    "ExecutionContext",
    "ExecutionResult",
    "ObservationType",
    "Observation",
    "LearnedPattern",
    "MemoryType",
    "PruningPolicy",
    # Classes
    "SkillRecorder",
    "SkillExecutor",
    "PatternLearner",
    "MemoryPruner",
]
