"""
Claude-First Aria - Voice-controlled computer use

This is the new Claude-first architecture where:
- Voice input via Whisper STT
- Claude Computer Use drives all actions
- Voice output via OpenAI TTS

Usage:
    python -m aria.claude_main
    python -m aria.claude_main "open safari and go to google.com"
"""

import sys
import time
import threading
import re
import subprocess
from typing import Optional, Tuple

from anthropic import Anthropic
from .claude_computer_use import create_agent, ClaudeComputerUseAgent
from .voice import VoiceInterface
from .control import get_control
from .config import PORCUPINE_ACCESS_KEY, ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MODEL_FAST


# Intent types
INTENT_FAST_COMMAND = "fast"      # Instant execution (open app, scroll, etc.)
INTENT_CONVERSATION = "conversation"  # Question/chat - no computer use needed
INTENT_COMPUTER_TASK = "computer"     # Needs screenshots and computer control


# Fast command patterns - these execute instantly without Claude
FAST_COMMANDS = {
    # Open apps: "open chrome", "open safari", "launch finder"
    r"^(?:open|launch|start)\s+(.+)$": "open_app",
    # Go to URL: "go to google.com", "navigate to youtube.com"
    r"^(?:go to|navigate to|open)\s+((?:https?://)?[\w.-]+\.[\w]+.*)$": "open_url",
    # Scroll: "scroll up", "scroll down"
    r"^scroll\s+(up|down)(?:\s+(\d+))?$": "scroll",
    # Type: "type hello world"
    r"^type\s+(.+)$": "type_text",
    # Press key: "press enter", "press escape"
    r"^press\s+(.+)$": "press_key",
    # Hotkey: "press command c", "do command shift s"
    r"^(?:press|do)\s+(command|cmd|ctrl|control|alt|option)\s+(.+)$": "hotkey",
}


class ClaudeAriaAgent:
    """
    Voice-controlled Claude Computer Use agent.

    Say a task, Claude does it autonomously.
    Now with intent classification for faster responses.
    """

    def __init__(self, voice_enabled: bool = True):
        self.voice = VoiceInterface() if voice_enabled else None
        self.control = get_control()
        self.is_running = False
        self._current_status = ""

        # Keep Claude client warm for fast conversation responses
        self._claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)

        # PRE-WARM the agent - create once, reuse for all tasks
        print("[Warming up Claude agent...]")
        self.agent = create_agent(
            on_message=self._on_message,
            on_action=self._on_action
        )
        print("[Agent ready!]")

        # Conversation patterns (questions that don't need computer control)
        self._conversation_patterns = [
            r"^(what|who|why|how|when|where|which|can you|could you|would you|do you|are you|is it|tell me)",
            r"(help|explain|describe|define|\?$)",
            r"^(hi|hello|hey|good morning|good afternoon|good evening)",
            r"^(yes|no|okay|ok|sure|thanks|thank you|please)",
        ]

    def _classify_intent(self, text: str) -> str:
        """
        Quickly classify user intent to route to the right handler.

        Returns:
            INTENT_FAST_COMMAND - for instant execution
            INTENT_CONVERSATION - for questions/chat (no computer use)
            INTENT_COMPUTER_TASK - for tasks needing screenshots/control
        """
        text_lower = text.lower().strip().rstrip('.!?,')

        # Check for fast commands first (instant)
        for pattern in FAST_COMMANDS.keys():
            if re.match(pattern, text_lower, re.IGNORECASE):
                return INTENT_FAST_COMMAND

        # Check for conversation patterns (questions, greetings, etc.)
        for pattern in self._conversation_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return INTENT_CONVERSATION

        # Keywords that indicate computer control is needed
        computer_keywords = [
            "click", "type", "scroll", "screenshot", "screen",
            "mouse", "keyboard", "window", "tab", "browser",
            "file", "folder", "desktop", "dock", "menu",
            "select", "copy", "paste", "drag", "move",
            "maximize", "minimize", "close", "resize",
        ]

        for keyword in computer_keywords:
            if keyword in text_lower:
                return INTENT_COMPUTER_TASK

        # Default to conversation for ambiguous inputs
        # This prevents unnecessary computer use for simple questions
        return INTENT_CONVERSATION

    def _handle_conversation(self, text: str) -> str:
        """
        Handle conversational queries quickly without computer use.
        Uses Claude directly for fast response.
        """
        print(f"[Conversation mode - fast response]")

        try:
            # Use Opus for conversation - maximum intelligence even for simple queries
            response = self._claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=300,  # Keep responses short for voice
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are Aria, a highly intelligent voice assistant with deep knowledge and reasoning ability. Respond briefly but thoughtfully (1-2 sentences max).

User said: "{text}"

Respond naturally:"""
                    }
                ]
            )

            result = response.content[0].text.strip()
            return result

        except Exception as e:
            print(f"[Conversation error: {e}]")
            return "I'm here and listening. What would you like me to do?"

    def _try_fast_command(self, text: str) -> Tuple[bool, str]:
        """
        Try to execute a simple command instantly without Claude.

        Returns:
            (success, response) - success=True if handled, response is what to say
        """
        # Clean up text - remove punctuation, extra spaces
        text_lower = text.lower().strip().rstrip('.!?,')

        for pattern, action in FAST_COMMANDS.items():
            match = re.match(pattern, text_lower, re.IGNORECASE)
            if match:
                try:
                    if action == "open_app":
                        app_name = match.group(1).strip()
                        # Map common names
                        app_map = {
                            "chrome": "Google Chrome",
                            "safari": "Safari",
                            "finder": "Finder",
                            "terminal": "Terminal",
                            "code": "Visual Studio Code",
                            "vscode": "Visual Studio Code",
                            "slack": "Slack",
                            "spotify": "Spotify",
                            "notes": "Notes",
                            "calendar": "Calendar",
                            "mail": "Mail",
                            "messages": "Messages",
                            "music": "Music",
                            "photos": "Photos",
                            "settings": "System Settings",
                            "preferences": "System Settings",
                        }
                        app = app_map.get(app_name.lower(), app_name.title())
                        subprocess.run(["open", "-a", app], check=True)
                        return True, f"Opened {app}."

                    elif action == "open_url":
                        url = match.group(1).strip()
                        if not url.startswith("http"):
                            url = "https://" + url
                        subprocess.run(["open", url], check=True)
                        return True, f"Opened {url}."

                    elif action == "scroll":
                        direction = match.group(1)
                        amount = int(match.group(2)) if match.group(2) else 3
                        scroll_amount = amount if direction == "up" else -amount
                        self.control.scroll(scroll_amount)
                        return True, f"Scrolled {direction}."

                    elif action == "type_text":
                        text_to_type = match.group(1)
                        self.control.type_text(text_to_type)
                        return True, "Typed it."

                    elif action == "press_key":
                        key = match.group(1).strip().lower()
                        self.control.press_key(key)
                        return True, f"Pressed {key}."

                    elif action == "hotkey":
                        modifier = match.group(1).lower()
                        key = match.group(2).strip().lower()
                        mod_map = {"cmd": "command", "ctrl": "control", "alt": "option"}
                        modifier = mod_map.get(modifier, modifier)
                        self.control.hotkey(modifier, key)
                        return True, "Done."

                except Exception as e:
                    print(f"[Fast command error: {e}]")
                    return False, ""  # Fall back to Claude

        return False, ""  # Not a fast command

    def _on_message(self, message: str):
        """Handle messages from Claude."""
        print(f"[Claude]: {message[:200]}..." if len(message) > 200 else f"[Claude]: {message}")
        self._current_status = message

    def _on_action(self, action: str):
        """Handle action notifications."""
        print(f"[Action]: {action}")

    def run_task(self, task: str) -> str:
        """
        Run a task using Claude Computer Use.

        Args:
            task: The task to perform (e.g., "open safari and go to google.com")

        Returns:
            The final response from Claude
        """
        print(f"\n{'='*50}")
        print(f"Task: {task}")
        print('='*50)

        # Speak that we're starting (ignore TTS errors)
        if self.voice:
            try:
                self.voice.speak("Working on it.")
            except Exception:
                pass  # TTS not available, continue silently

        # Run the PRE-WARMED agent (no cold start!)
        result = self.agent.run(task)

        print('='*50)
        print(f"Task complete!")

        return result

    def run_voice_loop(self):
        """
        Run the voice-controlled loop.

        Listen for voice commands, execute with Claude, speak results.
        """
        if not self.voice:
            print("Voice not available")
            return

        print("\n" + "="*50)
        print("Claude Aria - Voice Mode")
        print("="*50)
        print("Speak your commands. Say 'quit' or 'exit' to stop.")
        print("="*50 + "\n")

        self.is_running = True
        tts_available = True

        while self.is_running:
            try:
                # Listen for command
                if tts_available:
                    try:
                        self.voice.speak("Ready.")
                        # Wait for audio to fully stop to avoid feedback loop
                        time.sleep(0.5)
                    except Exception as e:
                        print(f"[TTS Error: {e}] - Continuing without voice output")
                        tts_available = False
                else:
                    print("\n[Ready - speak your command]")

                text = self.voice.listen(timeout=30.0)

                if not text:
                    continue

                text_lower = text.lower().strip()

                # Filter out audio feedback (mic picking up our own TTS)
                feedback_phrases = [
                    "ready", "working on it", "done", "goodbye", "yes",
                    "thank you", "thanks", "okay", "ok", "i see",
                    "got it", "alright", "sure", "right", "uh huh",
                ]
                if text_lower in feedback_phrases or len(text_lower) < 3:
                    print(f"[Filtered feedback: '{text}']")
                    continue

                # Filter phrases that are clearly not commands
                non_command_phrases = [
                    "thank you", "bye", "goodbye", "nice", "cool", "great",
                    "you guys are amazing", "that's amazing", "wow",
                ]
                if any(phrase in text_lower for phrase in non_command_phrases):
                    print(f"[Filtered non-command: '{text}']")
                    continue

                # Check for exit commands
                if text_lower in ["quit", "exit", "stop", "bye", "goodbye"]:
                    self.voice.speak("Goodbye!")
                    break

                # Check for status query
                if text_lower in ["status", "what are you doing", "progress"]:
                    self.voice.speak(self._current_status or "I'm ready for a task.")
                    continue

                # Classify intent for smart routing
                intent = self._classify_intent(text)
                print(f"[Intent: {intent}]")

                if intent == INTENT_FAST_COMMAND:
                    # Try fast command (instant execution)
                    handled, response = self._try_fast_command(text)
                    if handled:
                        print(f"[Fast]: {response}")
                        if tts_available:
                            try:
                                self.voice.speak(response)
                            except Exception:
                                tts_available = False
                        continue

                elif intent == INTENT_CONVERSATION:
                    # Handle conversation quickly without computer use
                    result = self._handle_conversation(text)
                    print(f"[Response]: {result}")
                    if tts_available:
                        try:
                            self.voice.speak(result)
                        except Exception:
                            tts_available = False
                            print(f"[Response]: {result}")
                    continue

                # INTENT_COMPUTER_TASK - Full Claude Computer Use
                result = self.run_task(text)

                # Speak a summary (if TTS available)
                if tts_available:
                    try:
                        if result:
                            short_result = result[:300] if len(result) > 300 else result
                            self.voice.speak(short_result)
                        else:
                            self.voice.speak("Done.")
                    except Exception:
                        tts_available = False
                        print(f"\n[Result]: {result[:500] if result else 'Done.'}")
                else:
                    print(f"\n[Result]: {result[:500] if result else 'Done.'}")

            except KeyboardInterrupt:
                print("\nInterrupted")
                break
            except Exception as e:
                print(f"Error: {e}")
                if self.voice:
                    self.voice.speak(f"Sorry, there was an error: {str(e)[:100]}")

        self.is_running = False
        print("Stopped.")

    def run_wake_word_loop(self):
        """
        Run with wake word detection.

        Say "Hey Aria" to activate, then give your command.
        """
        try:
            import pvporcupine
            import sounddevice as sd
            import numpy as np
        except ImportError:
            print("Wake word requires: pip install pvporcupine sounddevice")
            print("Falling back to continuous listening mode...")
            self.run_voice_loop()
            return

        if not PORCUPINE_ACCESS_KEY:
            print("PORCUPINE_ACCESS_KEY not set. Falling back to continuous mode...")
            self.run_voice_loop()
            return

        print("\n" + "="*50)
        print("Claude Aria - Wake Word Mode")
        print("="*50)
        print("Say 'Hey Aria' to activate, then give your command.")
        print("="*50 + "\n")

        # Initialize Porcupine
        porcupine = pvporcupine.create(
            access_key=PORCUPINE_ACCESS_KEY,
            keywords=["hey aria"]
        )

        self.is_running = True

        def audio_callback(indata, frames, time_info, status):
            if not self.is_running:
                return

            # Convert to int16
            audio_int16 = (indata.flatten() * 32767).astype(np.int16)

            # Check for wake word
            result = porcupine.process(audio_int16)
            if result >= 0:
                print("\n[Wake word detected!]")
                # Process in separate thread to not block audio
                threading.Thread(target=self._handle_wake, daemon=True).start()

        try:
            with sd.InputStream(
                samplerate=porcupine.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=porcupine.frame_length,
                callback=audio_callback
            ):
                print("Listening for 'Hey Aria'...")
                while self.is_running:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            porcupine.delete()
            self.is_running = False

    def _handle_wake(self):
        """Handle wake word activation."""
        if not self.voice:
            return

        self.voice.speak("Yes?")

        # Listen for command
        text = self.voice.listen(timeout=15.0)

        if not text:
            self.voice.speak("I didn't catch that.")
            return

        text_lower = text.lower().strip()

        # Check for exit
        if text_lower in ["quit", "exit", "stop"]:
            self.voice.speak("Goodbye!")
            self.is_running = False
            return

        # Run the task
        result = self.run_task(text)

        # Speak result
        if result:
            short_result = result[:300] if len(result) > 300 else result
            self.voice.speak(short_result)


    def run_text_loop(self):
        """
        Run text-input interactive mode.

        Type commands, Claude executes them.
        """
        print("\n" + "="*50)
        print("Claude Aria - Text Mode")
        print("="*50)
        print("Type your commands. Type 'quit' or 'exit' to stop.")
        print("="*50 + "\n")

        self.is_running = True

        while self.is_running:
            try:
                text = input("\n> ").strip()

                if not text:
                    continue

                text_lower = text.lower()

                if text_lower in ["quit", "exit", "stop", "bye"]:
                    print("Goodbye!")
                    break

                # Run the task
                result = self.run_task(text)
                print(f"\n[Result]: {result}")

            except KeyboardInterrupt:
                print("\nInterrupted")
                break
            except EOFError:
                break

        self.is_running = False
        print("Stopped.")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Claude Aria - Voice-controlled computer use")
    parser.add_argument("task", nargs="*", help="Task to execute (if not provided, enters interactive mode)")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice I/O")
    parser.add_argument("--text", action="store_true", help="Use text input mode instead of voice")
    parser.add_argument("--wake-word", action="store_true", help="Use wake word mode (say 'Hey Aria')")

    args = parser.parse_args()

    agent = ClaudeAriaAgent(voice_enabled=not args.no_voice and not args.text)

    if args.task:
        # Single task mode
        task = " ".join(args.task)
        result = agent.run_task(task)
        print(f"\nFinal result:\n{result}")
    elif args.text:
        # Text input mode
        agent.run_text_loop()
    elif args.wake_word:
        # Wake word mode
        agent.run_wake_word_loop()
    else:
        # Continuous voice mode
        agent.run_voice_loop()


if __name__ == "__main__":
    main()
