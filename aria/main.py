"""
Aria Agent - Main Entry Point

A menubar app that runs Aria as an always-on assistant.
"""

import json
import sys
import threading
import time
import traceback
from typing import Optional

import rumps

from .config import validate_config, OPENAI_API_KEY, REALTIME_VOICE_ENABLED
from .agent import get_agent
from .voice import get_voice, ConversationLoop, REALTIME_AVAILABLE
from .wake_word import create_wake_detector
from .vision import get_screen_capture

# Import realtime voice if available
if REALTIME_AVAILABLE:
    from .realtime_voice import (
        RealtimeVoiceClient,
        RealtimeConfig,
        RealtimeConversationLoop,
        ARIA_REALTIME_TOOLS,
        create_aria_tool_handler,
    )

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('aria')


class AriaMenubarApp(rumps.App):
    """Aria menubar application."""

    def __init__(self):
        super().__init__(
            "Aria",
            icon=None,  # Will use default
            quit_button=None  # Custom quit handling
        )

        self.agent = None
        self.voice = None
        self.wake_detector = None
        self.is_active = False
        self.is_listening = False

        # Menu items
        self.menu = [
            rumps.MenuItem("Activate (⌥ Space)", callback=self.on_activate),
            rumps.MenuItem("What's on screen?", callback=self.on_whats_on_screen),
            None,  # Separator
            rumps.MenuItem("Status: Idle", callback=None),
            None,  # Separator
            rumps.MenuItem("Preferences...", callback=self.on_preferences),
            rumps.MenuItem("Quit Aria", callback=self.on_quit),
        ]

        # Initialize in background
        threading.Thread(target=self._initialize, daemon=True).start()

    def _initialize(self):
        """Initialize Aria components."""
        logger.info("Starting Aria initialization...")

        # Check config
        missing = validate_config()
        if missing:
            logger.error(f"Missing config: {missing}")
            self._update_status(f"Missing: {', '.join(missing)}")
            rumps.notification(
                "Aria",
                "Configuration Error",
                f"Missing API keys: {', '.join(missing)}. Check .env file."
            )
            return

        try:
            # Initialize components
            logger.info("Initializing agent...")
            self.agent = get_agent()
            logger.info("Agent initialized")

            logger.info("Initializing voice...")
            self.voice = get_voice()
            logger.info("Voice initialized")

            # Set up wake word (optional - don't fail if it doesn't work)
            try:
                logger.info("Setting up wake word...")
                self.wake_detector = create_wake_detector(self.on_wake_word)
                self.wake_detector.start()
                logger.info("Wake word ready")
            except Exception as wake_err:
                logger.warning(f"Wake word setup failed (optional): {wake_err}")
                # Continue without wake word

            self._update_status("Ready")
            logger.info("Aria is ready!")
            rumps.notification(
                "Aria",
                "Ready",
                "Say 'Hey Aria' or press ⌥ Space to activate"
            )

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            logger.error(traceback.format_exc())
            self._update_status(f"Error: {str(e)[:30]}")
            rumps.notification(
                "Aria",
                "Error",
                f"Failed to initialize: {str(e)[:50]}"
            )

    def _update_status(self, status: str):
        """Update the status menu item."""
        for item in self.menu:
            if item and isinstance(item, rumps.MenuItem) and item.title.startswith("Status:"):
                item.title = f"Status: {status}"
                break

    def on_wake_word(self):
        """Called when wake word is detected."""
        if not self.is_active:
            self._start_conversation()

    @rumps.clicked("Activate (⌥ Space)")
    def on_activate(self, _):
        """Manual activation."""
        if not self.is_active:
            self._start_conversation()

    def _start_conversation(self):
        """Start a conversation turn."""
        if not self.agent or not self.voice:
            rumps.notification("Aria", "Not Ready", "Aria is still initializing...")
            return

        self.is_active = True
        self._update_status("Listening...")

        # Use Realtime Voice for natural, fluid conversation if available
        if REALTIME_AVAILABLE and REALTIME_VOICE_ENABLED:
            logger.info("Starting REALTIME conversation (low latency mode)")
            threading.Thread(target=self._realtime_conversation, daemon=True).start()
        else:
            # Fallback to traditional (slower) mode
            logger.info("Starting traditional conversation (higher latency)")
            threading.Thread(target=self._conversation_turn, daemon=True).start()

    def _conversation_turn(self):
        """Execute a continuous conversation until user ends it."""
        # Exit phrases that end the conversation
        EXIT_PHRASES = [
            "goodbye", "bye", "that's all", "thanks aria", "thank you aria",
            "stop", "quit", "exit", "done", "nevermind", "never mind",
            "go away", "dismiss", "that's it", "i'm done"
        ]

        try:
            # Acknowledge wake
            logger.info("Starting conversation")
            logger.info("About to speak greeting...")
            spoke = self.voice.speak("Hey! What can I help you with?")
            logger.info(f"Spoke greeting: {spoke}")
            time.sleep(0.2)

            turn_count = 0
            max_turns = 20  # Safety limit
            logger.info("Entering conversation loop...")

            while turn_count < max_turns:
                turn_count += 1

                # Listen for user
                self._update_status("Listening...")
                logger.info(f"Listening (turn {turn_count})...")
                user_input = self.voice.listen(timeout=30.0)  # Longer timeout

                if not user_input:
                    # No speech detected
                    if turn_count == 1:
                        self.voice.speak("I didn't catch that. What would you like?")
                        continue
                    else:
                        # After a conversation, silence means done
                        logger.info("Silence detected, ending conversation")
                        self.voice.speak("Let me know if you need anything else!")
                        break

                logger.info(f"User said: {user_input}")

                # Check for exit phrases
                user_lower = user_input.lower().strip()
                if any(phrase in user_lower for phrase in EXIT_PHRASES):
                    logger.info("Exit phrase detected")
                    self.voice.speak("Okay, talk to you later!")
                    break

                # Check if user said "Aria" again (re-activation, acknowledge)
                if user_lower in ["aria", "hey aria", "hi aria"]:
                    self.voice.speak("I'm here!")
                    continue

                self._update_status("Thinking...")

                # Decide if we need screen context
                # Skip screenshot for simple follow-ups to be faster
                needs_screen = self._needs_screen_context(user_input)

                logger.info(f"Getting response (screen: {needs_screen})...")
                response = self.agent.process_request(
                    user_input,
                    include_screen=needs_screen
                )
                logger.info(f"Response: {response[:100]}...")

                # Speak response
                self._update_status("Speaking...")
                self.voice.speak(response)
                time.sleep(0.3)  # Brief pause before listening again

            self._update_status("Ready")
            logger.info("Conversation ended")

        except Exception as e:
            logger.error(f"Conversation error: {e}")
            logger.error(traceback.format_exc())
            self._update_status("Ready")
            try:
                self.voice.speak("Sorry, something went wrong. Try again!")
            except:
                pass

        finally:
            self.is_active = False

    def _realtime_conversation(self):
        """Run a conversation using OpenAI Realtime API for natural, fluid voice."""
        import asyncio

        async def run_realtime():
            try:
                # Create realtime config with Aria's personality
                config = RealtimeConfig(
                    voice="shimmer",  # Natural, warm voice (nova not available in Realtime API)
                    vad_threshold=0.6,  # Slightly higher to reduce false triggers
                    silence_duration_ms=700,  # Wait a bit longer for natural pauses
                    instructions="""You are Aria, a friendly and intelligent AI assistant.

You have a warm, natural conversational style. Keep responses concise but helpful.
You can help with questions, control the computer, and have natural conversations.

Important:
- Be conversational and natural, not robotic
- Keep responses brief for voice (1-3 sentences unless explaining something)
- If asked to do something on the computer, explain what you'll do
- Remember context from the conversation"""
                )

                # Create tool handler that connects to Aria's agent
                def handle_tool(call_id: str, name: str, args: dict) -> str:
                    """Handle tool calls from realtime API."""
                    logger.info(f"Realtime tool call: {name}({args})")
                    try:
                        if name == "remember":
                            self.agent.memory.add_fact(args.get("fact", ""), args.get("category", "other"))
                            return '{"success": true}'
                        elif name == "recall":
                            results = self.agent.memory.search_memories(args.get("query", ""), n_results=5)
                            return json.dumps({"memories": results})
                        elif name == "click":
                            self.agent.control.click(args.get("x", 0), args.get("y", 0))
                            return '{"success": true}'
                        elif name == "type_text":
                            self.agent.control.type_text(args.get("text", ""))
                            return '{"success": true}'
                        elif name == "open_app":
                            self.agent.control.open_app(args.get("app", ""))
                            return '{"success": true}'
                        elif name == "open_url":
                            self.agent.control.open_url(args.get("url", ""))
                            return '{"success": true}'
                        else:
                            return f'{{"error": "Unknown tool: {name}"}}'
                    except Exception as e:
                        return f'{{"error": "{str(e)}"}}'

                # Create conversation loop
                loop = RealtimeConversationLoop(
                    api_key=OPENAI_API_KEY,
                    config=config,
                    tools=ARIA_REALTIME_TOOLS,
                    tool_handler=handle_tool
                )

                # Set up callbacks
                def on_user_speech(text):
                    logger.info(f"[User]: {text}")
                    self._update_status("Thinking...")

                def on_assistant_done(text):
                    logger.info(f"[Aria]: {text[:100]}...")
                    self._update_status("Listening...")

                loop.on_user_transcript = on_user_speech
                loop.on_assistant_done = on_assistant_done

                # Run the realtime conversation (blocking - runs until stopped)
                self._update_status("Listening...")
                logger.info("Realtime voice connected - speak naturally!")

                await loop.run()  # This blocks until conversation ends

            except Exception as e:
                logger.error(f"Realtime conversation error: {e}")
                logger.error(traceback.format_exc())
                # Fallback to traditional mode
                logger.info("Falling back to traditional voice mode")
                self._conversation_turn()

            finally:
                self.is_active = False
                self._update_status("Ready")

        # Run the async conversation
        try:
            asyncio.run(run_realtime())
        except Exception as e:
            logger.error(f"Realtime async error: {e}")
            self.is_active = False
            self._update_status("Ready")

    def _needs_screen_context(self, user_input: str) -> bool:
        """Determine if we need to capture screen for this request."""
        # Keywords that suggest we need to see the screen
        SCREEN_KEYWORDS = [
            "screen", "see", "look", "what's", "what is", "show", "click",
            "type", "open", "close", "window", "app", "button", "where",
            "find", "search", "this", "that", "here", "there", "current",
            # Action keywords - always need screen for these
            "scroll", "page", "go", "navigate", "facebook", "chrome",
            "safari", "browser", "website", "url", "tab", "menu"
        ]
        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in SCREEN_KEYWORDS)

    @rumps.clicked("What's on screen?")
    def on_whats_on_screen(self, _):
        """Quick action: describe what's on screen."""
        if not self.agent:
            return

        threading.Thread(target=self._describe_screen, daemon=True).start()

    def _describe_screen(self):
        """Describe current screen."""
        self._update_status("Looking...")
        try:
            description = self.agent.get_screen_context()
            if self.voice:
                self.voice.speak(description)
            self._update_status("Ready")
        except Exception as e:
            print(f"Screen description error: {e}")
            self._update_status("Error")

    @rumps.clicked("Preferences...")
    def on_preferences(self, _):
        """Open preferences (placeholder)."""
        rumps.notification(
            "Aria",
            "Preferences",
            "Preferences coming in v0.2. Edit .env file for now."
        )

    @rumps.clicked("Quit Aria")
    def on_quit(self, _):
        """Quit the app."""
        if self.wake_detector:
            self.wake_detector.stop()
        rumps.quit_application()


def main():
    """Main entry point."""
    print("=" * 50)
    print("   ARIA AGENT v0.1")
    print("=" * 50)

    # Check for required permissions
    print("\nRequired macOS Permissions:")
    print("  - Microphone (for voice)")
    print("  - Screen Recording (for screen capture)")
    print("  - Accessibility (for computer control)")
    print("\nGrant these in System Settings > Privacy & Security")
    print("=" * 50)
    print("\nStarting menubar app...")
    print("Aria will appear in your menubar (top-right).")
    print("Use the menubar icon or press ⌥ Space to activate.")
    print("\nPress Ctrl+C in this window to quit.")
    print("=" * 50 + "\n")

    try:
        # Run the app
        app = AriaMenubarApp()
        app.run()
    except KeyboardInterrupt:
        print("\nAria stopped by user.")
    except Exception as e:
        print(f"\nAria crashed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
