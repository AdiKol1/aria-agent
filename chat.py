#!/usr/bin/env python3
"""
Aria Chat - Interactive Text Mode

A simple way to interact with Aria using text input.
Uses the Qwen + Claude hybrid system for cost-effective responses.

Usage:
    python chat.py

Commands:
    - Type any command or question
    - Type 'quit' or 'q' to exit
    - Type 'help' to see examples
"""

import asyncio
import sys
import os

# Add aria to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment
from dotenv import load_dotenv
load_dotenv()

from aria.qwen_voice import QwenClaudeHybrid, QwenVoiceConfig


HELP_TEXT = """
╔══════════════════════════════════════════════════════════════╗
║                    ARIA CHAT - EXAMPLES                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  LOCAL COMMANDS (FREE, instant):                             ║
║    • open chrome         - Opens Google Chrome               ║
║    • open notes          - Opens Notes app                   ║
║    • scroll down         - Scrolls the page down             ║
║    • scroll up 5         - Scrolls up 5 times                ║
║    • type hello world    - Types "hello world"               ║
║    • press enter         - Presses Enter key                 ║
║    • copy / paste        - Clipboard operations              ║
║    • new tab             - Opens new browser tab             ║
║    • go to google.com    - Opens URL in browser              ║
║    • what time is it     - Shows current time                ║
║    • what's the date     - Shows current date                ║
║                                                              ║
║  CLAUDE QUERIES (~$0.01 each):                               ║
║    • why is the sky blue                                     ║
║    • explain recursion simply                                ║
║    • help me write a python function                         ║
║    • what do you think about...                              ║
║                                                              ║
║  CONTROLS:                                                   ║
║    • help   - Show this help                                 ║
║    • quit   - Exit the program                               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


async def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              ARIA CHAT - Hybrid Voice System                 ║")
    print("║                                                              ║")
    print("║  Simple commands → FREE (local)                              ║")
    print("║  Complex queries → Claude API (~$0.01)                       ║")
    print("║                                                              ║")
    print("║  Type 'help' for examples, 'quit' to exit                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Initialize the hybrid system
    config = QwenVoiceConfig()
    hybrid = QwenClaudeHybrid(config)

    # Show actions taken
    hybrid.on_action = lambda a: print(f"  ⚡ {a}")

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            # Handle special commands
            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break

            if user_input.lower() == 'help':
                print(HELP_TEXT)
                continue

            # Classify and show routing
            route, _ = hybrid.router.classify(user_input)
            cost = "FREE" if route == "local" else "~$0.01"
            print(f"  [{route.upper()}] {cost}")

            # Process and respond
            response = await hybrid.process_text(user_input)
            print(f"\nAria: {response}\n")

        except EOFError:
            print("\nGoodbye! 👋")
            break
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n  Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
