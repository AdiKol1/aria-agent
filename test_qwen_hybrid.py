#!/usr/bin/env python3
"""
Test script for Qwen + Claude Hybrid Voice System

This script tests both the routing logic and the actual execution
without requiring full voice input.
"""

import asyncio
import sys
import os
import urllib.request
import urllib.error

# Add aria to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

from aria.qwen_voice import QwenClaudeHybrid, TaskRouter, QwenVoiceConfig


def test_router():
    """Test the task router classification."""
    print("\n" + "=" * 60)
    print("TESTING TASK ROUTER")
    print("=" * 60)

    router = TaskRouter(QwenVoiceConfig())

    test_cases = [
        # Simple commands (should route to local)
        ("open chrome", "local"),
        ("open Safari", "local"),
        ("scroll down", "local"),
        ("scroll up 5", "local"),
        ("type hello world", "local"),
        ("press enter", "local"),
        ("copy", "local"),
        ("paste", "local"),
        ("new tab", "local"),
        ("go to google.com", "local"),
        ("what's the time", "local"),

        # Complex commands (should route to Claude)
        ("why is the sky blue", "claude"),
        ("explain how neural networks work", "claude"),
        ("help me write a python function", "claude"),
        ("analyze this code and find bugs", "claude"),
        ("first open chrome, then go to github", "claude"),
        ("what do you think about this approach", "claude"),
    ]

    passed = 0
    failed = 0

    for text, expected in test_cases:
        route, params = router.classify(text)
        status = "✅" if route == expected else "❌"
        if route == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} '{text}' → {route} (expected: {expected})")

    print(f"\nResults: {passed}/{passed + failed} passed")
    return failed == 0


async def test_local_commands():
    """Test local command execution."""
    print("\n" + "=" * 60)
    print("TESTING LOCAL COMMAND EXECUTION")
    print("=" * 60)

    # Check if we can import control module (requires display)
    try:
        import pyautogui
        pyautogui.size()  # This will fail if no display
        has_display = True
    except Exception as e:
        print(f"⚠️ Display/pyautogui not available: {e}")
        print("   Skipping local command tests (need GUI environment)")
        has_display = False
        return

    config = QwenVoiceConfig()
    config.use_local_for_simple = True

    hybrid = QwenClaudeHybrid(config)

    # Track actions
    actions = []
    hybrid.on_action = lambda a: actions.append(a)

    test_commands = [
        ("what's the time", "time"),
        ("what's the date", "date"),
        # Note: These will actually execute on your Mac!
        # Uncomment to test real actions:
        # ("open notes", "open_app"),
        # ("scroll down", "scroll"),
    ]

    print("\nTesting safe commands (no side effects):")
    for cmd, expected_type in test_commands:
        print(f"\n  Command: '{cmd}'")
        try:
            response = await hybrid.process_text(cmd)
            print(f"  Response: {response}")
        except Exception as e:
            print(f"  Error: {e}")


async def test_claude_routing():
    """Test that complex commands route to Claude."""
    print("\n" + "=" * 60)
    print("TESTING CLAUDE ROUTING")
    print("=" * 60)

    config = QwenVoiceConfig()
    hybrid = QwenClaudeHybrid(config)

    # This will actually call Claude API if ANTHROPIC_API_KEY is set
    test_commands = [
        "Why is the sky blue?",
        "Explain recursion in simple terms.",
    ]

    print("\nTesting Claude routing (requires ANTHROPIC_API_KEY):")
    for cmd in test_commands:
        print(f"\n  Command: '{cmd}'")
        try:
            response = await hybrid.process_text(cmd)
            print(f"  Response: {response[:200]}..." if len(response) > 200 else f"  Response: {response}")
        except Exception as e:
            print(f"  Error: {e}")
            print("  (This is expected if ANTHROPIC_API_KEY is not set)")


def test_ollama_available():
    """Check if Ollama is running."""
    print("\n" + "=" * 60)
    print("CHECKING OLLAMA STATUS")
    print("=" * 60)

    try:
        import urllib.request
        import json as json_module

        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json_module.loads(response.read().decode())
            models = data.get("models", [])
            print(f"✅ Ollama is running with {len(models)} model(s):")
            for model in models[:5]:
                print(f"   - {model.get('name')}")
            return True
    except urllib.error.URLError as e:
        print(f"❌ Ollama not running: {e.reason}")
        print("   Start with: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Ollama not running: {e}")
        print("   Start with: ollama serve")
        return False


async def interactive_test():
    """Run interactive test mode."""
    print("\n" + "=" * 60)
    print("INTERACTIVE TEST MODE")
    print("=" * 60)
    print("Type commands to test. Type 'quit' to exit.")
    print("Simple commands run locally, complex ones use Claude.")
    print("=" * 60)

    config = QwenVoiceConfig()
    hybrid = QwenClaudeHybrid(config)

    hybrid.on_action = lambda a: print(f"  [Action]: {a}")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            if not user_input:
                continue

            # Show routing decision
            route, params = hybrid.router.classify(user_input)
            print(f"  [Route]: {route}")

            response = await hybrid.process_text(user_input)
            print(f"Aria: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("QWEN + CLAUDE HYBRID VOICE SYSTEM TESTS")
    print("=" * 60)

    # Check environment
    print("\nChecking environment...")
    print(f"  ANTHROPIC_API_KEY: {'✅ Set' if os.getenv('ANTHROPIC_API_KEY') else '❌ Not set'}")
    print(f"  OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '⚠️ Not set (not required)'}")

    # Run tests
    test_router()
    test_ollama_available()
    await test_local_commands()

    # Only test Claude if API key is available
    if os.getenv("ANTHROPIC_API_KEY"):
        await test_claude_routing()
    else:
        print("\n⚠️ Skipping Claude tests (ANTHROPIC_API_KEY not set)")

    # Ask about interactive mode (skip if not interactive)
    print("\n" + "=" * 60)
    if sys.stdin.isatty():
        try:
            response = input("Run interactive test mode? [y/N]: ").strip().lower()
            if response == "y":
                await interactive_test()
        except EOFError:
            print("(Non-interactive mode - skipping interactive test)")
    else:
        print("(Non-interactive mode - skipping interactive test)")

    print("\n✅ All automated tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
