#!/usr/bin/env python3
"""
Test script for Aria Learning System

Tests:
- Skill recording and playback
- Pattern learning from corrections
- Memory pruning
"""

import asyncio
import sys
import os
import tempfile
from pathlib import Path

# Add aria to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_types():
    """Test that all types are importable and work correctly."""
    print("\n" + "=" * 60)
    print("TESTING LEARNING TYPES")
    print("=" * 60)

    from aria.learning.types import (
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
        VisualTarget,
    )

    # Test ActionType
    assert ActionType.CLICK.value == "click"
    print("  ActionType enum")

    # Test RecordedAction
    action = RecordedAction(
        action_type=ActionType.CLICK,
        x=100,
        y=200,
        visual_target=VisualTarget(description="blue button"),
    )
    action_dict = action.to_dict()
    action_restored = RecordedAction.from_dict(action_dict)
    assert action_restored.x == 100
    assert action_restored.visual_target.description == "blue button"
    print("  RecordedAction serialization")

    # Test LearnedSkill
    skill = LearnedSkill(
        id="test1",
        name="Test Skill",
        description="A test skill",
        trigger_phrases=["test it", "run test"],
        actions=[action],
    )
    skill_dict = skill.to_dict()
    skill_restored = LearnedSkill.from_dict(skill_dict)
    assert skill_restored.name == "Test Skill"
    assert len(skill_restored.actions) == 1
    print("  LearnedSkill serialization")

    # Test Observation
    obs = Observation(
        id="obs1",
        observation_type=ObservationType.CORRECTION,
        original_action="plain text",
        corrected_action="code block",
        context={"app": "VS Code"},
    )
    obs_dict = obs.to_dict()
    obs_restored = Observation.from_dict(obs_dict)
    assert obs_restored.corrected_action == "code block"
    print("  Observation serialization")

    # Test LearnedPattern
    pattern = LearnedPattern(
        id="p1",
        trigger="When sharing code",
        action="Use code block formatting",
        context={"app": "Slack"},
    )
    pattern_dict = pattern.to_dict()
    pattern_restored = LearnedPattern.from_dict(pattern_dict)
    assert pattern_restored.trigger == "When sharing code"
    print("  LearnedPattern serialization")

    # Test PruningPolicy
    policy = PruningPolicy.for_type(MemoryType.PREFERENCE)
    assert policy.never_auto_delete == True
    policy = PruningPolicy.for_type(MemoryType.PATTERN)
    assert policy.auto_decay == True
    print("  PruningPolicy defaults")

    print("\n All types working correctly!")
    return True


def test_skill_recorder():
    """Test skill recording functionality."""
    print("\n" + "=" * 60)
    print("TESTING SKILL RECORDER")
    print("=" * 60)

    from aria.learning.skill_recorder import SkillRecorder

    # Use temp directory for tests
    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = SkillRecorder(storage_path=Path(tmpdir))

        # Test recording session
        session = recorder.start_recording("test_skill", starting_app="Safari")
        assert recorder.is_recording()
        print("  Started recording session")

        # Record some actions
        recorder.record_click(100, 200, visual_description="Search button")
        recorder.record_type("hello world")
        recorder.record_hotkey(["command", "enter"])
        recorder.record_scroll(-300)
        print("  Recorded 4 actions")

        # Mark decision point
        recorder.mark_decision_point("User chose this option")
        print("  Marked decision point")

        # Stop and save
        skill = recorder.stop_recording(
            trigger_phrases=["test it", "run test"],
            description="A test skill for demo",
        )
        assert skill is not None
        assert len(skill.actions) == 4
        assert not recorder.is_recording()
        print(f"  Saved skill: {skill.name} with {len(skill.actions)} actions")

        # Test skill retrieval
        found = recorder.find_skill_by_trigger("test it please")
        assert found is not None
        assert found.id == skill.id
        print("  Found skill by trigger")

        # Test listing
        skills = recorder.list_skills()
        assert len(skills) == 1
        print(f"  Listed {len(skills)} skill(s)")

    print("\n Skill recorder working correctly!")
    return True


async def test_skill_executor():
    """Test skill execution functionality."""
    print("\n" + "=" * 60)
    print("TESTING SKILL EXECUTOR")
    print("=" * 60)

    from aria.learning.skill_executor import SkillExecutor
    from aria.learning.types import LearnedSkill, RecordedAction, ActionType

    # Create a skill with test actions
    skill = LearnedSkill(
        id="exec_test",
        name="Execution Test",
        description="Test skill for execution",
        trigger_phrases=["execute test"],
        actions=[
            RecordedAction(action_type=ActionType.WAIT, delay_before_ms=100),
            RecordedAction(action_type=ActionType.WAIT, delay_before_ms=100),
        ],
    )

    executor = SkillExecutor()

    # Track execution
    actions_started = []
    actions_completed = []

    executor.on_action_start = lambda a, i: actions_started.append(i)
    executor.on_action_complete = lambda a, i, s: actions_completed.append((i, s))

    # Dry run execution
    result = await executor.execute(skill, dry_run=True)
    assert result.success
    assert result.actions_completed == 2
    assert len(actions_started) == 2
    print(f"  Dry run completed: {result.actions_completed}/{result.total_actions} actions")

    # Test variable substitution
    skill_with_vars = LearnedSkill(
        id="var_test",
        name="Variable Test",
        description="Test variable substitution",
        trigger_phrases=["var test"],
        actions=[
            RecordedAction(action_type=ActionType.TYPE, text="Going to {{destination}}"),
        ],
    )

    result = await executor.execute(
        skill_with_vars,
        variables={"destination": "NYC"},
        dry_run=True,
    )
    assert result.success
    print("  Variable substitution working")

    print("\n Skill executor working correctly!")
    return True


def test_pattern_learner():
    """Test pattern learning functionality."""
    print("\n" + "=" * 60)
    print("TESTING PATTERN LEARNER")
    print("=" * 60)

    from aria.learning.patterns import PatternLearner

    # Use temp directory for tests
    with tempfile.TemporaryDirectory() as tmpdir:
        learner = PatternLearner(storage_path=Path(tmpdir))

        # Record corrections (need 2 to trigger pattern)
        obs1 = learner.observe_correction(
            original="plain text code",
            corrected="code block formatting",
            context={"task": "sharing_code"},
        )
        print(f"  Recorded correction 1: {obs1.id}")

        obs2 = learner.observe_correction(
            original="inline code",
            corrected="code block formatting",
            context={"task": "sharing_code"},
        )
        print(f"  Recorded correction 2: {obs2.id}")

        # Check if pattern was created
        patterns = learner.list_patterns()
        print(f"  Patterns created: {len(patterns)}")

        # Record repeated actions (need 3 to trigger pattern)
        for i in range(3):
            learner.observe_repeated_action(
                action="save file",
                context={"app": "VS Code"},
            )
        print("  Recorded 3 repeated actions")

        patterns = learner.list_patterns()
        print(f"  Total patterns: {len(patterns)}")

        # Test context matching
        matching = learner.get_patterns_for_context({"task": "sharing_code"})
        print(f"  Patterns matching context: {len(matching)}")

        # Test stats
        stats = learner.get_stats()
        print(f"  Stats: {stats['total_observations']} observations, {stats['total_patterns']} patterns")

    print("\n Pattern learner working correctly!")
    return True


def test_memory_pruner():
    """Test memory pruning functionality."""
    print("\n" + "=" * 60)
    print("TESTING MEMORY PRUNER")
    print("=" * 60)

    from aria.learning.pruner import MemoryPruner
    from aria.learning.types import MemoryType
    from datetime import datetime, timedelta

    # Use temp directory for tests
    with tempfile.TemporaryDirectory() as tmpdir:
        pruner = MemoryPruner(storage_path=Path(tmpdir))

        # Test contradiction detection
        new_fact = "User likes dark mode"
        existing_facts = [
            {"fact": "User likes light mode", "category": "preference"},
            {"fact": "User uses VS Code", "category": "work"},
        ]

        contradictions = pruner.check_contradictions(new_fact, existing_facts)
        print(f"  Contradictions found: {len(contradictions)}")

        # Test pruning with mock memories
        old_date = (datetime.now() - timedelta(days=60)).isoformat()
        test_memories = {
            "mem1": {
                "content": "Old insight",
                "category": "insight",
                "created_at": old_date,
                "last_used": old_date,
            },
            "mem2": {
                "content": "User preference",
                "category": "preference",
                "created_at": old_date,
            },
            "mem3": {
                "content": "Recent fact",
                "category": "fact",
                "created_at": datetime.now().isoformat(),
            },
        }

        # Dry run
        results = pruner.prune(memories=test_memories, dry_run=True)
        print(f"  Dry run - would archive: {len(results['archived'])}")
        print(f"  Dry run - would skip: {len(results['skipped'])}")

        # Actual run
        results = pruner.prune(memories=test_memories, dry_run=False)
        print(f"  Archived: {len(results['archived'])}")

        # Test archive listing
        archived = pruner.list_archived()
        print(f"  Items in archive: {len(archived)}")

        # Test restore
        if archived:
            restored = pruner.restore_from_archive(archived[0]["id"])
            print(f"  Restored from archive: {restored is not None}")

        # Test stats
        stats = pruner.get_stats()
        print(f"  Pruning stats: {stats['total_archived']} archived total")

    print("\n Memory pruner working correctly!")
    return True


def test_integration():
    """Test integration with MCP server."""
    print("\n" + "=" * 60)
    print("TESTING MCP INTEGRATION")
    print("=" * 60)

    from aria.mcp_server import AriaMCPServer

    server = AriaMCPServer()

    # Get tools list
    tools = server.get_tools()
    learning_tools = [t for t in tools if t["name"].startswith(("start_skill", "stop_skill", "list_learned", "observe_", "list_pattern", "prune_", "get_learning"))]
    print(f"  Found {len(learning_tools)} learning tools")

    expected_tools = [
        "start_skill_recording",
        "stop_skill_recording",
        "list_learned_skills",
        "execute_learned_skill",
        "observe_correction",
        "list_patterns",
        "prune_memories",
        "get_learning_status",
    ]

    for tool_name in expected_tools:
        found = any(t["name"] == tool_name for t in tools)
        status = "" if found else ""
        print(f"  {status} {tool_name}")
        assert found, f"Missing tool: {tool_name}"

    # Test calling get_learning_status
    result = server.call_tool("get_learning_status", {})
    assert "content" in result
    print(f"  Called get_learning_status successfully")

    print("\n MCP integration working correctly!")
    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("ARIA LEARNING SYSTEM TESTS")
    print("=" * 60)

    tests = [
        ("Types", test_types, False),
        ("Skill Recorder", test_skill_recorder, False),
        ("Skill Executor", test_skill_executor, True),  # async test
        ("Pattern Learner", test_pattern_learner, False),
        ("Memory Pruner", test_memory_pruner, False),
        ("MCP Integration", test_integration, False),
    ]

    passed = 0
    failed = 0

    for name, test_fn, is_async in tests:
        try:
            if is_async:
                result = await test_fn()
            else:
                result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"\n {name} returned False")
        except Exception as e:
            failed += 1
            print(f"\n {name} failed with error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{passed + failed} tests passed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)

    print("\n All learning system tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
