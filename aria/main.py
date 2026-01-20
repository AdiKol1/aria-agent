"""
Aria Agent - Main Entry Point

A menubar app that runs Aria as an always-on assistant.

# HYBRID VOICE ARCHITECTURE:
# - Voice I/O: OpenAI Realtime API (230ms latency) for STT/TTS
# - Brain: Claude Opus 4.5 for complex reasoning and understanding
# - Computer Use: Claude Computer Use agent for mouse/keyboard/screen tasks
#
# This gives us the best of both worlds:
# - Fast, natural voice interactions via Realtime API
# - Claude's superior intelligence for complex tasks and reasoning
# - Seamless computer control through Claude Computer Use
#
# Routing Logic:
# 1. Voice input captured via OpenAI Realtime (fast STT)
# 2. Task classification determines routing:
#    - Computer control tasks -> Claude Computer Use agent
#    - Simple Q&A/conversation -> Can use faster local responses
#    - Complex reasoning -> Claude Opus 4.5
# 3. Response spoken via OpenAI Realtime (fast TTS)
"""

import json
import sys
import threading
import time
import traceback
from typing import Optional

import rumps

# Configuration imports
# REALTIME_VOICE_ENABLED: Enable OpenAI Realtime API for fast voice I/O (230ms latency)
# GEMINI_VOICE_ENABLED: Enable Gemini Live API for native multimodal voice
#
# For hybrid architecture, we want REALTIME_VOICE_ENABLED=True for fast STT/TTS,
# while the "brain" (Claude) handles complex reasoning and computer control tasks.
# Set via environment: REALTIME_VOICE_ENABLED=true/false
from .config import (
    validate_config, OPENAI_API_KEY, GOOGLE_API_KEY,
    REALTIME_VOICE_ENABLED, GEMINI_VOICE_ENABLED,
    GEMINI_VOICE_MODEL, GEMINI_VOICE_VOICE, GEMINI_SAMPLE_RATE, GEMINI_OUTPUT_SAMPLE_RATE
)
from .agent import get_agent
from .voice import get_voice, ConversationLoop, REALTIME_AVAILABLE
from .wake_word import create_wake_detector
from .vision import get_screen_capture
from .action_executor import VisionActionExecutor, get_action_executor

# Import Claude Computer Use for smart computer control
# In the hybrid architecture, this is the "brain" for complex computer tasks.
# While Realtime API handles fast voice I/O, Claude Computer Use handles:
# - Mouse/keyboard control with vision verification
# - Multi-step computer tasks with planning
# - Complex reasoning about screen content
try:
    from .claude_computer_use import create_agent as create_claude_agent
    CLAUDE_COMPUTER_USE_AVAILABLE = True
except ImportError:
    CLAUDE_COMPUTER_USE_AVAILABLE = False
    create_claude_agent = None

# Gemini voice - lazy import for fast startup
# Set to None initially; will be checked lazily when needed
GENAI_AVAILABLE = None  # Will be set on first check
_gemini_imports = None

def _load_gemini():
    """Lazy load Gemini voice module."""
    global GENAI_AVAILABLE, _gemini_imports
    if GENAI_AVAILABLE is None:
        try:
            from .gemini_voice import (
                GeminiVoiceClient,
                GeminiVoiceConfig,
                GeminiConversationLoop,
                ARIA_GEMINI_TOOLS,
                create_gemini_tool_handler,
                GENAI_AVAILABLE as _GENAI_AVAIL,
            )
            GENAI_AVAILABLE = _GENAI_AVAIL
            _gemini_imports = {
                'GeminiVoiceClient': GeminiVoiceClient,
                'GeminiVoiceConfig': GeminiVoiceConfig,
                'GeminiConversationLoop': GeminiConversationLoop,
                'ARIA_GEMINI_TOOLS': ARIA_GEMINI_TOOLS,
                'create_gemini_tool_handler': create_gemini_tool_handler,
            }
        except ImportError:
            GENAI_AVAILABLE = False
    return GENAI_AVAILABLE

# Import realtime voice if available
# HYBRID ARCHITECTURE: OpenAI Realtime API provides ultra-low latency (~230ms)
# voice I/O (STT/TTS). This is the "ears and mouth" of the hybrid system,
# while Claude serves as the "brain" for complex reasoning and computer control.
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

        # Claude Computer Use agent (pre-warmed for fast response)
        self.claude_agent = None

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

            # Pre-warm Claude Computer Use agent for fast response
            if CLAUDE_COMPUTER_USE_AVAILABLE:
                logger.info("Pre-warming Claude Computer Use agent...")
                self.claude_agent = create_claude_agent(
                    on_message=lambda msg: logger.info(f"[Claude]: {msg[:100]}..."),
                    on_action=lambda action: logger.info(f"[Action]: {action}")
                )
                logger.info("Claude agent ready!")
            else:
                logger.warning("Claude Computer Use not available")

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

        # HYBRID VOICE ARCHITECTURE - Voice Mode Selection
        # =================================================
        # We use a layered approach for optimal latency and intelligence:
        #
        # Option 1: Gemini Live API (if available and enabled)
        #   - Native multimodal voice with good reasoning
        #   - Lower latency than traditional STT+LLM+TTS pipeline
        #
        # Option 2: OpenAI Realtime API (PREFERRED for hybrid architecture)
        #   - Ultra-low latency (~230ms) for voice I/O (STT/TTS)
        #   - Computer control tasks are routed to Claude Computer Use
        #   - Simple Q&A can use Realtime's fast responses
        #   - Complex reasoning routed to Claude Opus 4.5
        #
        # Option 3: Traditional mode (fallback)
        #   - Sequential: Whisper STT -> Claude -> TTS
        #   - Higher latency but works without Realtime/Gemini APIs
        #
        # The key insight: Realtime API gives us fast voice I/O, while
        # Claude handles the actual "thinking" for complex tasks.

        if _load_gemini() and GEMINI_VOICE_ENABLED:
            logger.info("Starting GEMINI conversation (Gemini 2.5 Flash + Live API)")
            threading.Thread(target=self._gemini_conversation, daemon=True).start()
        elif REALTIME_AVAILABLE and REALTIME_VOICE_ENABLED:
            # HYBRID MODE: Realtime for voice I/O, Claude for computer tasks
            logger.info("Starting HYBRID conversation (OpenAI Realtime voice + Claude brain)")
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

                # SMART ROUTING: Use Claude Computer Use for tasks needing control
                needs_computer = self._needs_computer_control(user_input)

                if needs_computer and self.claude_agent:
                    # Use Claude Computer Use for tasks needing mouse/keyboard/screenshots
                    logger.info(f"Routing to Claude Computer Use...")
                    response = self._run_claude_task(user_input)
                else:
                    # Use fast conversational response for questions/chat
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

    def _gemini_conversation(self):
        """Run a conversation using Gemini 3 Flash + Live API for superior reasoning."""
        import asyncio

        async def run_gemini():
            try:
                # ===== KNOWLEDGE INJECTION: Load learned context =====
                startup_context = ""
                try:
                    startup_context = self.agent.memory.get_startup_context()
                    if startup_context:
                        logger.info(f"[Knowledge] Injected {len(startup_context)} chars of learned context")
                except Exception as ctx_err:
                    logger.warning(f"[Knowledge] Failed to load context: {ctx_err}")

                # Build dynamic system instructions with injected knowledge
                base_instructions = """You are Aria, an intelligent voice assistant that controls a Mac computer.

#########################################################
## RULE #0: ABSOLUTE HONESTY - THE MOST IMPORTANT RULE ##
#########################################################

YOU MUST NEVER LIE OR CONFABULATE. This is non-negotiable.

People depend on you. A personal assistant that lies is DANGEROUS.
Claiming to do something you didn't do can cause real harm.

ABSOLUTE RULES:
1. NEVER say "Done", "Opening", "Clicking", "Typing" unless you ACTUALLY called a tool
2. NEVER claim success before you receive a tool result
3. If you didn't call a tool, you CANNOT claim to have done the action
4. If a tool fails, say "That didn't work" - don't pretend it succeeded
5. If unsure, say "I'm not sure if that worked, can you check?"

BEFORE speaking any confirmation, ask yourself:
- Did I actually call a tool? (Not just think about it - ACTUALLY call it)
- Did the tool return success:true?
- Only then can you say "Done"

If you find yourself saying "I'll open/click/type..." - STOP.
Instead, CALL THE TOOL FIRST, then confirm AFTER.

#########################################################

## NO CODE EXECUTION
NEVER use Python code execution. It doesn't work.
ONLY use the provided function tools (open_app, click, type_text, etc.)

## BE CONCISE
You are a VOICE assistant. Keep responses SHORT (1-2 sentences).
Don't explain - DO IT and confirm briefly.
Good: [call tool] "Done."
Bad: "I'll open Chrome for you now. Let me use the open_app function..."

## USE TOOLS FOR ACTIONS
When asked to DO something, call the appropriate tool:
- "Open Chrome" -> call open_app(app="Google Chrome")
- "Click button" -> call click(target="the button")
- "Type hello" -> call type_text(text="hello")

## YOU CAN SEE THE SCREEN
You receive screenshots (1280 x 828 pixels). You CAN see what's on screen.
Your coordinates will be automatically SCALED to the actual screen.

## HOW TO CLICK/MOVE MOUSE

**PREFERRED for web content, buttons, links, icons, text:**
- click_target(target="description") - Claude's vision finds the element precisely and clicks it
- move_to_target(target="description") - Claude's vision finds the element and moves mouse there
- Examples:
  - click_target(target="the blue Send button")
  - click_target(target="the user's name in top right")
  - move_to_target(target="Adi Kol in the top right corner")
  - move_to_target(target="the search input field")
This is MORE ACCURATE than coordinates because Claude analyzes the screen.

**For DOCK ICONS only (bottom of screen):**
- click_element(element="Chrome") - click dock icon by name
- move_mouse_to_element(element="Finder") - move mouse to dock icon

**For precise pixel positions (when you know exact coords):**
- click_at_coordinates(x, y) - click at specific screen position
- move_mouse_to_coordinates(x, y) - move mouse to specific position

## HOW TO FILL FORM FIELDS
Use fill_field(field="description", text="value") to click a field AND type in it.
Examples:
- fill_field(field="First Name", text="John")
- fill_field(field="Email input", text="test@example.com")
- fill_field(field="search box", text="my search query")

## BROWSER TAB CONTROL (MUCH MORE RELIABLE than keyboard shortcuts)
- "Close this tab" -> close_browser_tab() - closes current tab
- "Close the Gmail tab" -> close_browser_tab(tab="Gmail")
- "Close second tab from right" -> close_tab_by_position(position="second from right")
- "Switch to YouTube" -> switch_browser_tab(tab="YouTube")
- "What tabs are open?" -> list_browser_tabs()

## KEYBOARD SHORTCUTS (for non-tab actions)
- "Hit enter" / "Press enter" -> press_key(key="return")
- "Undo" -> hotkey(keys=["command", "z"])
- "Copy" -> hotkey(keys=["command", "c"])
- "Paste" -> hotkey(keys=["command", "v"])
- "New tab" -> hotkey(keys=["command", "t"])

## SCROLL COMMANDS
Use scroll(amount) for scrolling. Negative = down, Positive = up.
- "Scroll down" -> scroll(amount=-300)
- "Scroll up" -> scroll(amount=300)
- "Scroll down a lot" -> scroll(amount=-600)
- "Scroll 30" or "Scroll 30%" -> scroll(amount=-300)  (treat as scroll down)
- Just "Scroll" -> scroll(amount=-300)  (default to scroll down)
DO NOT ask "how much" - just use a reasonable default (300 pixels).

## YOUR TOOLS (YOU MUST USE THESE)
DOCK ONLY: move_mouse_to_element(element), click_element(element) - ONLY for dock icons at screen bottom!
EVERYTHING ELSE: click_at_coordinates(x, y), move_mouse_to_coordinates(x, y) - For web content, text, buttons, names!
BROWSER TABS: list_browser_tabs(), close_browser_tab(tab), switch_browser_tab(tab), close_tab_by_position(position) - MUCH BETTER than Cmd+W!
FORMS: fill_field(field, text) - Click AND type in one action
KEYBOARD: hotkey(keys), type_text(text), press_key(key)
APPS: open_app(app), open_url(url), scroll(amount)
MEMORY: remember(fact), recall(query)

## THE USER
- The user's name is KOL (pronounced "coal")

## KEY BEHAVIORS
1. Be BRIEF - short sentences only
2. ONLY say "Done" AFTER a tool call returns success:true
3. NEVER claim you did something if you didn't call a tool
4. If you're unsure, ask "Did that work?" instead of claiming success
5. Don't narrate your actions - call the tool, THEN confirm
6. Never use code execution - only use tool functions"""

                # Inject learned context into instructions
                if startup_context:
                    full_instructions = base_instructions + "\n\n## YOUR LEARNED KNOWLEDGE (from previous sessions)\n" + startup_context
                else:
                    full_instructions = base_instructions

                # Create Gemini config with dynamically injected knowledge
                GeminiVoiceConfig = _gemini_imports['GeminiVoiceConfig']
                config = GeminiVoiceConfig(
                    model=GEMINI_VOICE_MODEL,
                    voice=GEMINI_VOICE_VOICE,
                    sample_rate=GEMINI_SAMPLE_RATE,
                    output_sample_rate=GEMINI_OUTPUT_SAMPLE_RATE,
                    instructions=full_instructions
                )

                # Create vision-guided action executor
                action_executor = get_action_executor(self.agent.control)

                # Action deduplication - prevent same action within cooldown period
                recent_actions = {}  # {action_key: timestamp}
                ACTION_COOLDOWN = 5.0  # seconds

                # Create tool handler with vision-guided actions
                def handle_tool(call_id: str, name: str, args: dict) -> str:
                    """Handle tool calls from Gemini API with vision guidance."""
                    logger.info(f"Gemini tool call: {name}({args})")
                    logger.info(f"[DEBUG] Handler entered for: {name}")

                    # DEDUPLICATION: Check if this action was recently executed
                    # Only dedupe actions that have side effects (open_app, click, type, etc.)
                    dedupe_actions = ["open_app", "click", "click_at_coordinates", "type_text",
                                      "fill_field", "press_key", "hotkey", "scroll"]
                    if name in dedupe_actions:
                        logger.info(f"[DEBUG] Checking dedupe for: {name}")
                        action_key = f"{name}:{json.dumps(args, sort_keys=True)}"
                        now = time.time()
                        logger.info(f"[DEBUG] action_key={action_key}, recent_actions keys={list(recent_actions.keys())}")
                        if action_key in recent_actions:
                            last_time = recent_actions[action_key]
                            if now - last_time < ACTION_COOLDOWN:
                                elapsed = now - last_time
                                logger.info(f"[DEDUPE] SKIPPED duplicate {name}({args}) - executed {elapsed:.1f}s ago")
                                return json.dumps({
                                    "success": True,
                                    "deduplicated": True,
                                    "message": f"Action already completed {elapsed:.1f}s ago. No need to repeat."
                                })
                        # Mark action as executed
                        recent_actions[action_key] = now
                        logger.info(f"[DEBUG] Marked action as executed: {action_key}")

                    try:
                        logger.info(f"[DEBUG] Entering try block for: {name}")
                        logger.info(f"[DEBUG] name type: {type(name)}, name repr: {repr(name)}")

                        # TRACE: Log each branch check
                        logger.info(f"[DEBUG] Checking branches... name='{name}'")

                        if name == "remember":
                            logger.info(f"[DEBUG] MATCHED: remember")
                            fact = args.get("fact", "")
                            category = args.get("category", "other")
                            print(f"[Memory] Remembering: {fact[:100]} (category: {category})")
                            result = self.agent.memory.remember_fact(fact, category)
                            print(f"[Memory] Remember result: {result}")
                            return json.dumps({"success": result, "fact": fact})

                        elif name == "recall":
                            query = args.get("query", "")
                            print(f"[Memory] Recalling: {query}")
                            results = self.agent.memory.search_memories(query, n_results=5)
                            print(f"[Memory] Found {len(results)} memories")
                            if results:
                                for i, mem in enumerate(results):
                                    print(f"[Memory]   {i+1}. {mem[:80]}...")
                            return json.dumps({"success": True, "query": query, "memories": results, "count": len(results)})

                        elif name == "look_at_screen":
                            # Screen images are sent to Gemini continuously
                            # This tool returns screen metadata - Gemini can SEE the actual screen
                            focus = args.get("focus", "")
                            print(f"[Vision] Looking at screen" + (f" (focus: {focus})" if focus else ""))

                            # Get screen info and ensure fresh capture is sent
                            result = action_executor._capture_screen_b64(use_cache=False)
                            if not result:
                                return json.dumps({"error": "Could not capture screen"})

                            image_b64, (width, height) = result
                            active_app = self.agent.control.get_frontmost_app()

                            # Return metadata - Gemini sees the actual screen in its context
                            return json.dumps({
                                "screen_size": [width, height],
                                "active_app": active_app,
                                "focus": focus if focus else "full screen",
                                "instruction": "You can now see the screen (updated every 2 seconds). Look at the current screen image, identify the x,y coordinates of what you want to click, then use click_at_coordinates(x=X, y=Y)."
                            })

                        elif name == "click":
                            # If target description provided, use vision to find element
                            target = args.get("target", "")
                            if target:
                                result = action_executor.click_element(target)
                                if result.success:
                                    return json.dumps({"success": True, "message": result.message})
                                else:
                                    # Element not found - guide to use coordinates instead
                                    return json.dumps({
                                        "success": False,
                                        "message": f"Could not find '{target}' on screen",
                                        "suggestion": "Use look_at_screen to see the screen, identify the x,y coordinates of the element, then use click_at_coordinates(x=X, y=Y) instead."
                                    })
                            else:
                                # Blind click at coordinates (not recommended)
                                x, y = args.get("x", 0), args.get("y", 0)
                                self.agent.control.click(x, y)
                                return '{"success": true, "warning": "Clicked without vision verification"}'

                        elif name == "execute_task":
                            # High-level task with vision planning and verification
                            task = args.get("task", "")
                            result = action_executor.execute_task(task)
                            return json.dumps({
                                "success": result.success,
                                "message": result.message,
                                "details": result.details
                            })

                        elif name == "open_menu_item":
                            menu = args.get("menu", "")
                            item = args.get("item", "")
                            result = action_executor.open_menu_item(menu, item)
                            return json.dumps({"success": result.success, "message": result.message})

                        elif name == "type_text":
                            logger.info(f"[DEBUG] MATCHED: type_text")
                            text = args.get("text", "")

                            # SAFEGUARD: Detect if Gemini is trying to type user feedback OR its own questions
                            feedback_patterns = [
                                # User feedback patterns
                                "you didn't", "you did not", "you still", "you just",
                                "didn't do it", "not going to be sent", "didn't hit send",
                                "didn't click", "didn't type", "still didn't", "what's the issue",
                                "work correctly", "do it already", "see you do", "not seeing you",
                                "that's not", "you're wrong", "try again", "do it right",
                                "follow the steps", "i asked you", "i told you",
                                # Aria's own clarifying questions - NEVER type these
                                "what should i", "what do you want", "could you tell me",
                                "please specify", "can you clarify", "what exactly",
                                "which one", "how much", "what would you like",
                                "should i ask", "what to type", "what message"
                            ]
                            text_lower = text.lower()
                            is_feedback = any(pattern in text_lower for pattern in feedback_patterns)

                            if is_feedback:
                                print(f"[BLOCKED] Gemini tried to type user feedback as content: '{text[:80]}...'")
                                return json.dumps({
                                    "success": False,
                                    "error": "BLOCKED: This looks like user feedback, not message content. Ask the user what they want to type."
                                })

                            print(f"[Action] Typing text: '{text[:50]}...'")
                            self.agent.control.type_text(text)
                            return '{"success": true}'

                        elif name == "fill_field":
                            logger.info(f"[DEBUG] MATCHED: fill_field")
                            # Combined click + type for form fields
                            field = args.get("field", "")
                            text = args.get("text", "")

                            # SAFEGUARD: Detect if Gemini is trying to type user feedback OR its own questions
                            feedback_patterns = [
                                # User feedback patterns
                                "you didn't", "you did not", "you still", "you just",
                                "didn't do it", "not going to be sent", "didn't hit send",
                                "didn't click", "didn't type", "still didn't", "what's the issue",
                                "work correctly", "do it already", "see you do", "not seeing you",
                                "that's not", "you're wrong", "try again", "do it right",
                                # Aria's own clarifying questions - NEVER type these
                                "what should i", "what do you want", "could you tell me",
                                "please specify", "can you clarify", "what exactly",
                                "which one", "how much", "what would you like",
                                "should i ask", "what to type", "what message"
                            ]
                            text_lower = text.lower()
                            is_feedback = any(pattern in text_lower for pattern in feedback_patterns)

                            if is_feedback:
                                print(f"[BLOCKED] Gemini tried to fill field with user feedback: '{text[:80]}...'")
                                return json.dumps({
                                    "success": False,
                                    "error": "BLOCKED: This looks like user feedback, not message content. Ask the user what they want to type."
                                })

                            print(f"[Action] Fill field: '{field}' with text: '{text}'")

                            # First, click on the field using vision
                            click_result = action_executor.click_element(field)
                            if not click_result.success:
                                return json.dumps({
                                    "success": False,
                                    "error": f"Could not find field: {field}",
                                    "message": click_result.message
                                })

                            # Small delay to ensure field is focused
                            time.sleep(0.2)

                            # Now type the text
                            self.agent.control.type_text(text)
                            print(f"[Action] Typed '{text}' into '{field}'")

                            return json.dumps({
                                "success": True,
                                "field": field,
                                "text": text,
                                "message": f"Filled '{field}' with '{text}'"
                            })

                        elif name == "hotkey":
                            logger.info(f"[DEBUG] MATCHED: hotkey")
                            keys = args.get("keys", [])
                            print(f"[Action] Hotkey: {keys}")
                            if keys:
                                result = self.agent.control.hotkey(*keys)
                                print(f"[Action] Hotkey result: {result}")
                                return json.dumps({"success": result, "keys": keys})
                            else:
                                print("[Action] Hotkey failed - no keys provided")
                                return json.dumps({"success": False, "error": "No keys provided"})

                        elif name == "press_key":
                            logger.info(f"[DEBUG] MATCHED: press_key")
                            self.agent.control.press_key(args.get("key", ""))
                            return '{"success": true}'

                        elif name == "open_app":
                            logger.info(f"[DEBUG] MATCHED: open_app")
                            app_name = args.get("app", "")

                            # Fix common speech recognition errors
                            app_corrections = {
                                "cloud": "Claude",
                                "clawed": "Claude",
                                "claud": "Claude",
                                "claw": "Claude",
                                "chrome": "Google Chrome",
                                "crome": "Google Chrome",
                                "krome": "Google Chrome",
                            }
                            app_lower = app_name.lower()
                            if app_lower in app_corrections:
                                corrected = app_corrections[app_lower]
                                logger.info(f"[Action] Correcting '{app_name}' -> '{corrected}'")
                                app_name = corrected

                            logger.info(f"[Action] Opening app: {app_name}")
                            try:
                                result = self.agent.control.open_app(app_name)
                                logger.info(f"[Action] open_app returned: {result}")
                                if result:
                                    logger.info(f"[Action] Successfully opened {app_name}")
                                    return json.dumps({"success": True, "message": f"Opened {app_name}"})
                                else:
                                    logger.info(f"[Action] Failed to open {app_name} - app not found?")
                                    return json.dumps({"success": False, "error": f"Could not open '{app_name}'. App may not exist."})
                            except Exception as open_err:
                                logger.error(f"[Action] open_app failed: {open_err}")
                                return json.dumps({"success": False, "error": str(open_err)})

                        elif name == "open_url":
                            url = args.get("url", "")
                            # Add https:// if no protocol specified
                            if url and not url.startswith(("http://", "https://")):
                                url = "https://" + url
                            success = self.agent.control.open_url(url)
                            if success:
                                return json.dumps({"success": True, "url": url})
                            else:
                                return json.dumps({"success": False, "error": f"Failed to open {url}"})

                        elif name == "scroll":
                            logger.info(f"[DEBUG] MATCHED: scroll")
                            amount = args.get("amount", 0)
                            logger.info(f"[Action] Scroll: {amount}")
                            result = self.agent.control.scroll(amount)
                            logger.info(f"[Action] Scroll result: {result}")
                            return json.dumps({"success": result, "amount": amount})

                        elif name == "web_search":
                            query = args.get("query", "")
                            print(f"[Action] Web search: {query}")
                            try:
                                # Use Gemini with Google Search grounding
                                from google import genai
                                from google.genai import types as genai_types
                                search_client = genai.Client(api_key=GOOGLE_API_KEY)

                                # Create the search request with grounding
                                response = search_client.models.generate_content(
                                    model="gemini-2.0-flash-exp",
                                    contents=f"Search the web and provide a concise answer to: {query}",
                                    config=genai_types.GenerateContentConfig(
                                        tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())]
                                    )
                                )

                                result_text = response.text if response.text else "No results found"
                                print(f"[Action] Web search result: {result_text[:200]}...")
                                return json.dumps({"success": True, "query": query, "result": result_text})
                            except Exception as search_err:
                                print(f"[Action] Web search error: {search_err}")
                                return json.dumps({"success": False, "error": str(search_err)})

                        elif name == "double_click":
                            target = args.get("target", "")
                            print(f"[Action] Double-click: {target}")
                            element = action_executor.find_element(target)
                            if element:
                                x, y = element["x"], element["y"]
                                print(f"[Action] Found at ({x}, {y}), double-clicking")
                                result = self.agent.control.double_click(x, y)
                                return json.dumps({"success": result, "target": target, "x": x, "y": y})
                            return json.dumps({"success": False, "error": f"Could not find: {target}"})

                        elif name == "right_click":
                            target = args.get("target", "")
                            print(f"[Action] Right-click: {target}")
                            element = action_executor.find_element(target)
                            if element:
                                x, y = element["x"], element["y"]
                                print(f"[Action] Found at ({x}, {y}), right-clicking")
                                result = self.agent.control.right_click(x, y)
                                return json.dumps({"success": result, "target": target, "x": x, "y": y})
                            return json.dumps({"success": False, "error": f"Could not find: {target}"})

                        elif name == "move_mouse":
                            logger.info(f"[DEBUG] MATCHED: move_mouse")
                            target = args.get("target", "")
                            logger.info(f"[Action] Move mouse to: {target}")
                            element = action_executor.find_element(target)
                            logger.info(f"[Action] find_element returned: {element}")
                            if element:
                                x, y = element["x"], element["y"]
                                logger.info(f"[Action] Moving mouse to ({x}, {y})")
                                result = self.agent.control.move_to(x, y)
                                logger.info(f"[Action] move_to result: {result}")
                                return json.dumps({"success": result, "target": target, "x": x, "y": y})
                            logger.info(f"[Action] Could not find element: {target}")
                            return json.dumps({"success": False, "error": f"Could not find: {target}"})

                        elif name == "drag":
                            from_target = args.get("from_target", "")
                            to_target = args.get("to_target", "")
                            print(f"[Action] Drag from '{from_target}' to '{to_target}'")
                            from_element = action_executor.find_element(from_target)
                            to_element = action_executor.find_element(to_target)
                            if from_element and to_element:
                                start_x, start_y = from_element["x"], from_element["y"]
                                end_x, end_y = to_element["x"], to_element["y"]
                                print(f"[Action] Dragging from ({start_x}, {start_y}) to ({end_x}, {end_y})")
                                result = self.agent.control.drag_to(start_x, start_y, end_x, end_y)
                                return json.dumps({"success": result, "from": [start_x, start_y], "to": [end_x, end_y]})
                            error = f"Could not find: {from_target if not from_element else to_target}"
                            return json.dumps({"success": False, "error": error})

                        elif name == "get_mouse_position":
                            import pyautogui
                            x, y = pyautogui.position()
                            print(f"[Action] Mouse position: ({x}, {y})")
                            return json.dumps({"x": x, "y": y, "screen_size": [self.agent.control.screen_width, self.agent.control.screen_height]})

                        elif name == "click_at_coordinates":
                            x = args.get("x", 0)
                            y = args.get("y", 0)
                            button = args.get("button", "left")

                            # Scale coordinates from screenshot space to actual screen space
                            # Gemini sees screenshots at SCREENSHOT_MAX_WIDTH, so ALWAYS scale if screen is larger
                            from .config import SCREENSHOT_MAX_WIDTH
                            screen_width = self.agent.control.screen_width
                            screen_height = self.agent.control.screen_height

                            if screen_width > SCREENSHOT_MAX_WIDTH:
                                scale = screen_width / SCREENSHOT_MAX_WIDTH
                                scaled_x = int(x * scale)
                                scaled_y = int(y * scale)
                                logger.info(f"[Action] Click at ({x}, {y}) -> scaled to ({scaled_x}, {scaled_y}) [scale: {scale:.2f}x] with {button} button")
                            else:
                                scaled_x = int(x)
                                scaled_y = int(y)
                                logger.info(f"[Action] Click at ({x}, {y}) (no scaling needed) with {button} button")

                            result = self.agent.control.click(scaled_x, scaled_y, button=button)
                            return json.dumps({"success": result, "x": scaled_x, "y": scaled_y, "button": button})

                        elif name == "move_mouse_to_coordinates":
                            # FAST direct coordinate movement - no vision lookup needed
                            x = args.get("x", 0)
                            y = args.get("y", 0)

                            # Scale coordinates from screenshot space to actual screen space
                            from .config import SCREENSHOT_MAX_WIDTH
                            screen_width = self.agent.control.screen_width
                            screen_height = self.agent.control.screen_height

                            # ALWAYS scale if screen is larger than screenshot
                            # Gemini sees screenshots at SCREENSHOT_MAX_WIDTH, so coordinates are in that space
                            if screen_width > SCREENSHOT_MAX_WIDTH:
                                scale = screen_width / SCREENSHOT_MAX_WIDTH
                                scaled_x = int(x * scale)
                                scaled_y = int(y * scale)
                                logger.info(f"[Action] Move mouse to ({x}, {y}) -> scaled to ({scaled_x}, {scaled_y}) [scale: {scale:.2f}x]")
                            else:
                                scaled_x = int(x)
                                scaled_y = int(y)
                                logger.info(f"[Action] Move mouse to ({x}, {y}) (no scaling needed)")

                            result = self.agent.control.move_to(scaled_x, scaled_y)
                            return json.dumps({"success": result, "x": scaled_x, "y": scaled_y})

                        elif name == "move_mouse_to_element":
                            # Use Accessibility API to find element and move mouse
                            from .accessibility import find_ui_element
                            element = args.get("element", "")
                            logger.info(f"[Action] Move mouse to element: {element}")

                            result = find_ui_element(element)
                            if result:
                                x, y = result["center_x"], result["center_y"]
                                logger.info(f"[Action] Found {result['name']} at ({x}, {y})")
                                self.agent.control.move_to(x, y)
                                return json.dumps({
                                    "success": True,
                                    "element": result["name"],
                                    "x": x,
                                    "y": y
                                })
                            else:
                                logger.info(f"[Action] Element not found: {element}")
                                return json.dumps({
                                    "success": False,
                                    "error": f"Could not find element: {element}"
                                })

                        elif name == "click_element":
                            # Use Accessibility API to find element and click
                            from .accessibility import find_ui_element
                            element = args.get("element", "")
                            logger.info(f"[Action] Click element: {element}")

                            result = find_ui_element(element)
                            if result:
                                x, y = result["center_x"], result["center_y"]
                                logger.info(f"[Action] Found {result['name']} at ({x}, {y}), clicking...")
                                self.agent.control.click(x, y)
                                return json.dumps({
                                    "success": True,
                                    "element": result["name"],
                                    "x": x,
                                    "y": y,
                                    "action": "clicked"
                                })
                            else:
                                logger.info(f"[Action] Element not found: {element}")
                                return json.dumps({
                                    "success": False,
                                    "error": f"Could not find element: {element}"
                                })

                        elif name == "click_target":
                            # Use Claude's vision to find element precisely and click
                            target = args.get("target", "")
                            button = args.get("button", "left")
                            logger.info(f"[Action] Click target (Claude vision): '{target}'")

                            # Use action_executor's find_element which uses Claude's vision
                            element = action_executor.find_element(target)
                            if element:
                                x, y = element["x"], element["y"]
                                logger.info(f"[Action] Claude found '{target}' at ({x}, {y}), clicking with {button} button")
                                result = self.agent.control.click(x, y, button=button)
                                return json.dumps({
                                    "success": result,
                                    "target": target,
                                    "x": x,
                                    "y": y,
                                    "button": button,
                                    "method": "claude_vision"
                                })
                            else:
                                logger.info(f"[Action] Claude could not find: '{target}'")
                                return json.dumps({
                                    "success": False,
                                    "error": f"Could not find '{target}' on screen. Try a more specific description."
                                })

                        elif name == "move_to_target":
                            # Use Claude's vision to find element precisely and move mouse there
                            target = args.get("target", "")
                            logger.info(f"[Action] Move to target (Claude vision): '{target}'")

                            # Use action_executor's find_element which uses Claude's vision
                            element = action_executor.find_element(target)
                            if element:
                                x, y = element["x"], element["y"]
                                logger.info(f"[Action] Claude found '{target}' at ({x}, {y}), moving mouse")
                                result = self.agent.control.move_to(x, y)
                                return json.dumps({
                                    "success": result,
                                    "target": target,
                                    "x": x,
                                    "y": y,
                                    "method": "claude_vision"
                                })
                            else:
                                logger.info(f"[Action] Claude could not find: '{target}'")
                                return json.dumps({
                                    "success": False,
                                    "error": f"Could not find '{target}' on screen. Try a more specific description."
                                })

                        elif name == "list_browser_tabs":
                            # List all browser tabs using AppleScript
                            from .accessibility import list_browser_tabs
                            browser = args.get("browser", "Google Chrome")
                            logger.info(f"[Action] Listing {browser} tabs")
                            result = list_browser_tabs(browser)
                            return json.dumps({
                                "success": True,
                                "tabs": result
                            })

                        elif name == "close_browser_tab":
                            # Close a browser tab using AppleScript
                            from .accessibility import close_browser_tab
                            tab = args.get("tab", None)  # None = close current tab
                            browser = args.get("browser", "Google Chrome")
                            logger.info(f"[Action] Closing tab: {tab if tab else 'current'} in {browser}")
                            result = close_browser_tab(tab, browser)
                            logger.info(f"[Action] Close tab result: {result}")
                            return json.dumps(result)

                        elif name == "switch_browser_tab":
                            # Switch to a browser tab using AppleScript
                            from .accessibility import switch_browser_tab
                            tab = args.get("tab", "")
                            browser = args.get("browser", "Google Chrome")
                            logger.info(f"[Action] Switching to tab: {tab} in {browser}")
                            result = switch_browser_tab(tab, browser)
                            logger.info(f"[Action] Switch tab result: {result}")
                            return json.dumps(result)

                        elif name == "close_tab_by_position":
                            # Close a browser tab by position using AppleScript
                            from .accessibility import close_tab_by_position
                            position = args.get("position", "")
                            browser = args.get("browser", "Google Chrome")
                            logger.info(f"[Action] Closing tab by position: {position} in {browser}")
                            result = close_tab_by_position(position, browser)
                            logger.info(f"[Action] Close tab result: {result}")
                            return json.dumps(result)

                        elif name == "schedule_research":
                            # Schedule proactive learning on a topic
                            from .learning import get_learning_engine
                            topic = args.get("topic", "")
                            query = args.get("query", "")
                            interval = args.get("interval_hours", 168)  # Default: weekly

                            print(f"[Learning] Scheduling research: {topic}")
                            learning = get_learning_engine()
                            success = learning.schedule_research(topic, query, interval)

                            if success:
                                return json.dumps({
                                    "success": True,
                                    "message": f"Scheduled proactive research on '{topic}' every {interval} hours",
                                    "topic": topic
                                })
                            else:
                                return json.dumps({
                                    "success": False,
                                    "error": f"Research on '{topic}' is already scheduled"
                                })

                        elif name == "list_research":
                            # List all scheduled research tasks
                            from .learning import get_learning_engine
                            learning = get_learning_engine()
                            tasks = learning.list_research_tasks()

                            print(f"[Learning] Listing {len(tasks)} research tasks")
                            return json.dumps({
                                "success": True,
                                "research_tasks": tasks,
                                "count": len(tasks)
                            })

                        else:
                            logger.info(f"[DEBUG] UNMATCHED tool: {name}")
                            return f'{{"error": "Unknown tool: {name}"}}'

                    except Exception as e:
                        logger.error(f"[DEBUG] EXCEPTION in tool handler: {e}")
                        import traceback
                        traceback.print_exc()
                        return f'{{"error": "{str(e)}"}}'

                # Create conversation loop
                GeminiConversationLoop = _gemini_imports['GeminiConversationLoop']
                ARIA_GEMINI_TOOLS = _gemini_imports['ARIA_GEMINI_TOOLS']
                loop = GeminiConversationLoop(
                    api_key=GOOGLE_API_KEY,
                    config=config,
                    tools=ARIA_GEMINI_TOOLS,
                    tool_handler=handle_tool
                )

                # SHARE DEDUP STATE: Make VoiceBridge use the same recent_actions dict
                # as the tool handler so they don't duplicate actions
                loop.client._recent_actions = recent_actions
                logger.info("[DEDUP] Shared recent_actions between tool handler and VoiceBridge")

                # Set up reconnection callback for user feedback
                def on_reconnecting(attempt: int, delay: float):
                    logger.info(f"[Reconnect] Attempt {attempt}, waiting {delay:.1f}s...")
                    self._update_status(f"Reconnecting ({attempt})...")
                    if attempt == 1:
                        rumps.notification(
                            "Aria",
                            "Connection Lost",
                            "Reconnecting to voice service..."
                        )

                loop.on_reconnecting = on_reconnecting

                # ===== AUTOMATIC LEARNING SYSTEM =====
                # Track each conversation turn for automatic knowledge extraction
                turn_state = {
                    "user_input": "",
                    "tool_calls": [],  # List of {name, args, result}
                }

                # Wrap the tool handler to track tool calls AND record outcomes
                original_handle_tool = handle_tool
                def tracking_handle_tool(call_id: str, name: str, args: dict) -> str:
                    result = original_handle_tool(call_id, name, args)

                    # Track this tool call for learning
                    turn_state["tool_calls"].append({
                        "name": name,
                        "args": args,
                        "result": result[:500] if len(result) > 500 else result
                    })

                    # ===== ACTION OUTCOME TRACKING =====
                    # Record success/failure to learn what works
                    try:
                        result_data = json.loads(result)
                        # Determine success from result
                        success = result_data.get("success", True)  # Default to success if not specified
                        if "error" in result_data:
                            success = False

                        # Record the outcome (in background to not slow down)
                        def record():
                            try:
                                context = f"user_goal: {turn_state.get('user_input', '')[:100]}"
                                self.agent.memory.record_action_outcome(
                                    action_name=name,
                                    action_args=args,
                                    success=success,
                                    context=context
                                )
                            except Exception:
                                pass  # Don't let tracking errors affect execution

                        threading.Thread(target=record, daemon=True).start()
                    except (json.JSONDecodeError, AttributeError):
                        pass  # Can't parse result, skip tracking

                    return result

                # Use tracking handler
                loop.tool_handler = tracking_handle_tool
                loop.client.on_tool_call = loop._handle_tool_call  # Re-wire with new handler

                # Set up callbacks with automatic learning
                def on_user_speech(text):
                    logger.info(f"[User]: {text}")
                    self._update_status("Thinking...")
                    # Start new turn - store user input
                    turn_state["user_input"] = text
                    turn_state["tool_calls"] = []  # Reset tool calls for new turn

                def on_assistant_done(text):
                    logger.info(f"[Aria]: {text[:100]}...")
                    self._update_status("Listening...")

                    # ===== AUTOMATIC LEARNING: Extract and store knowledge =====
                    # Run in background to not block the conversation
                    user_input = turn_state.get("user_input", "")
                    tool_calls = turn_state.get("tool_calls", [])

                    if user_input and text:  # Only learn from meaningful turns
                        def extract_learnings():
                            try:
                                logger.info(f"[Learning] Extracting knowledge from turn...")
                                self.agent.memory.extract_and_store_memories(
                                    user_input=user_input,
                                    assistant_response=text,
                                    actions_taken=tool_calls
                                )
                                logger.info(f"[Learning] Knowledge extraction complete")
                            except Exception as learn_err:
                                logger.warning(f"[Learning] Failed to extract: {learn_err}")

                        # Run in background thread
                        threading.Thread(target=extract_learnings, daemon=True).start()

                    # Reset for next turn
                    turn_state["user_input"] = ""
                    turn_state["tool_calls"] = []

                loop.on_user_transcript = on_user_speech
                loop.on_assistant_done = on_assistant_done

                # Run the Gemini conversation (blocking - runs until stopped)
                self._update_status("Listening...")
                logger.info("Gemini 3 Flash voice connected - speak naturally!")

                await loop.run()  # This blocks until conversation ends

            except Exception as e:
                logger.error(f"Gemini conversation error: {e}")
                logger.error(traceback.format_exc())
                # Fallback to OpenAI Realtime or traditional mode
                if REALTIME_AVAILABLE and REALTIME_VOICE_ENABLED:
                    logger.info("Falling back to OpenAI Realtime voice mode")
                    self._realtime_conversation()
                else:
                    logger.info("Falling back to traditional voice mode")
                    self._conversation_turn()

            finally:
                self.is_active = False
                self._update_status("Ready")

        # Run the async conversation
        try:
            asyncio.run(run_gemini())
        except Exception as e:
            logger.error(f"Gemini async error: {e}")
            self.is_active = False
            self._update_status("Ready")

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
                    instructions="""You are Aria, a friendly and intelligent AI assistant with FULL VISION of the user's screen.

You have a warm, natural conversational style. Keep responses concise but helpful.
You can see the screen, control the computer, and have natural conversations.

## YOU CAN SEE THE SCREEN
YES, you have vision! Call **look_at_screen** to see what's on the user's display.
- ALWAYS call look_at_screen when asked "what's on my screen?", "can you see this?", or similar
- Call it before any visual task to see the current state
- You can focus on specific areas like "the menu bar" or "any error messages"

## How to Control the Computer

1. **look_at_screen** - See what's currently on screen. Call this FIRST for any visual question or task.

2. **execute_task** - PREFERRED for actions. Describe what you want: "open a new Chrome window". The system sees the screen and figures out how.

3. **click** - Click by DESCRIPTION: "the File menu", "the blue button". NEVER guess coordinates.

4. **open_menu_item** - For menu actions: specify menu and item names.

5. **open_app** - Opens apps by name.

## Important Guidelines:
- You CAN see the screen - use look_at_screen!
- ALWAYS use description-based tools, NEVER guess coordinates
- The system verifies actions succeeded before confirming
- Be conversational and natural
- Keep responses brief for voice (1-3 sentences)
- After actions, briefly confirm what happened"""
                )

                # Create vision-guided action executor
                action_executor = get_action_executor(self.agent.control)

                # Create tool handler with vision-guided actions
                def handle_tool(call_id: str, name: str, args: dict) -> str:
                    """Handle tool calls from realtime API with vision guidance."""
                    logger.info(f"Realtime tool call: {name}({args})")
                    try:
                        if name == "remember":
                            fact = args.get("fact", "")
                            category = args.get("category", "other")
                            print(f"[Memory] Remembering: {fact[:100]} (category: {category})")
                            result = self.agent.memory.remember_fact(fact, category)
                            print(f"[Memory] Remember result: {result}")
                            return json.dumps({"success": result, "fact": fact})

                        elif name == "recall":
                            query = args.get("query", "")
                            print(f"[Memory] Recalling: {query}")
                            results = self.agent.memory.search_memories(query, n_results=5)
                            print(f"[Memory] Found {len(results)} memories")
                            if results:
                                for i, mem in enumerate(results):
                                    print(f"[Memory]   {i+1}. {mem[:80]}...")
                            return json.dumps({"success": True, "query": query, "memories": results, "count": len(results)})

                        elif name == "look_at_screen":
                            # Screen images are sent to Gemini continuously
                            # This tool returns screen metadata - Gemini can SEE the actual screen
                            focus = args.get("focus", "")
                            print(f"[Vision] Looking at screen" + (f" (focus: {focus})" if focus else ""))

                            # Get screen info and ensure fresh capture is sent
                            result = action_executor._capture_screen_b64(use_cache=False)
                            if not result:
                                return json.dumps({"error": "Could not capture screen"})

                            image_b64, (width, height) = result
                            active_app = self.agent.control.get_frontmost_app()

                            # Return metadata - Gemini sees the actual screen in its context
                            return json.dumps({
                                "screen_size": [width, height],
                                "active_app": active_app,
                                "focus": focus if focus else "full screen",
                                "instruction": "You can now see the screen (updated every 2 seconds). Look at the current screen image, identify the x,y coordinates of what you want to click, then use click_at_coordinates(x=X, y=Y)."
                            })

                        elif name == "click":
                            # If target description provided, use vision to find element
                            target = args.get("target", "")
                            if target:
                                result = action_executor.click_element(target)
                                if result.success:
                                    return json.dumps({"success": True, "message": result.message})
                                else:
                                    # Element not found - guide to use coordinates instead
                                    return json.dumps({
                                        "success": False,
                                        "message": f"Could not find '{target}' on screen",
                                        "suggestion": "Use look_at_screen to see the screen, identify the x,y coordinates of the element, then use click_at_coordinates(x=X, y=Y) instead."
                                    })
                            else:
                                # Blind click at coordinates (not recommended)
                                x, y = args.get("x", 0), args.get("y", 0)
                                self.agent.control.click(x, y)
                                return '{"success": true, "warning": "Clicked without vision verification"}'

                        elif name == "execute_task":
                            # High-level task with vision planning and verification
                            task = args.get("task", "")
                            result = action_executor.execute_task(task)
                            return json.dumps({
                                "success": result.success,
                                "message": result.message,
                                "details": result.details
                            })

                        elif name == "open_menu_item":
                            menu = args.get("menu", "")
                            item = args.get("item", "")
                            result = action_executor.open_menu_item(menu, item)
                            return json.dumps({"success": result.success, "message": result.message})

                        elif name == "type_text":
                            logger.info(f"[DEBUG] MATCHED: type_text")
                            text = args.get("text", "")

                            # SAFEGUARD: Detect if Gemini is trying to type user feedback OR its own questions
                            feedback_patterns = [
                                # User feedback patterns
                                "you didn't", "you did not", "you still", "you just",
                                "didn't do it", "not going to be sent", "didn't hit send",
                                "didn't click", "didn't type", "still didn't", "what's the issue",
                                "work correctly", "do it already", "see you do", "not seeing you",
                                "that's not", "you're wrong", "try again", "do it right",
                                "follow the steps", "i asked you", "i told you",
                                # Aria's own clarifying questions - NEVER type these
                                "what should i", "what do you want", "could you tell me",
                                "please specify", "can you clarify", "what exactly",
                                "which one", "how much", "what would you like",
                                "should i ask", "what to type", "what message"
                            ]
                            text_lower = text.lower()
                            is_feedback = any(pattern in text_lower for pattern in feedback_patterns)

                            if is_feedback:
                                print(f"[BLOCKED] Gemini tried to type user feedback as content: '{text[:80]}...'")
                                return json.dumps({
                                    "success": False,
                                    "error": "BLOCKED: This looks like user feedback, not message content. Ask the user what they want to type."
                                })

                            print(f"[Action] Typing text: '{text[:50]}...'")
                            self.agent.control.type_text(text)
                            return '{"success": true}'

                        elif name == "fill_field":
                            logger.info(f"[DEBUG] MATCHED: fill_field")
                            # Combined click + type for form fields
                            field = args.get("field", "")
                            text = args.get("text", "")

                            # SAFEGUARD: Detect if Gemini is trying to type user feedback OR its own questions
                            feedback_patterns = [
                                # User feedback patterns
                                "you didn't", "you did not", "you still", "you just",
                                "didn't do it", "not going to be sent", "didn't hit send",
                                "didn't click", "didn't type", "still didn't", "what's the issue",
                                "work correctly", "do it already", "see you do", "not seeing you",
                                "that's not", "you're wrong", "try again", "do it right",
                                # Aria's own clarifying questions - NEVER type these
                                "what should i", "what do you want", "could you tell me",
                                "please specify", "can you clarify", "what exactly",
                                "which one", "how much", "what would you like",
                                "should i ask", "what to type", "what message"
                            ]
                            text_lower = text.lower()
                            is_feedback = any(pattern in text_lower for pattern in feedback_patterns)

                            if is_feedback:
                                print(f"[BLOCKED] Gemini tried to fill field with user feedback: '{text[:80]}...'")
                                return json.dumps({
                                    "success": False,
                                    "error": "BLOCKED: This looks like user feedback, not message content. Ask the user what they want to type."
                                })

                            print(f"[Action] Fill field: '{field}' with text: '{text}'")

                            # First, click on the field using vision
                            click_result = action_executor.click_element(field)
                            if not click_result.success:
                                return json.dumps({
                                    "success": False,
                                    "error": f"Could not find field: {field}",
                                    "message": click_result.message
                                })

                            # Small delay to ensure field is focused
                            time.sleep(0.2)

                            # Now type the text
                            self.agent.control.type_text(text)
                            print(f"[Action] Typed '{text}' into '{field}'")

                            return json.dumps({
                                "success": True,
                                "field": field,
                                "text": text,
                                "message": f"Filled '{field}' with '{text}'"
                            })

                        elif name == "hotkey":
                            logger.info(f"[DEBUG] MATCHED: hotkey")
                            keys = args.get("keys", [])
                            print(f"[Action] Hotkey: {keys}")
                            if keys:
                                result = self.agent.control.hotkey(*keys)
                                print(f"[Action] Hotkey result: {result}")
                                return json.dumps({"success": result, "keys": keys})
                            else:
                                print("[Action] Hotkey failed - no keys provided")
                                return json.dumps({"success": False, "error": "No keys provided"})

                        elif name == "press_key":
                            logger.info(f"[DEBUG] MATCHED: press_key")
                            self.agent.control.press_key(args.get("key", ""))
                            return '{"success": true}'

                        elif name == "open_app":
                            logger.info(f"[DEBUG] MATCHED: open_app")
                            app_name = args.get("app", "")

                            # Fix common speech recognition errors
                            app_corrections = {
                                "cloud": "Claude",
                                "clawed": "Claude",
                                "claud": "Claude",
                                "claw": "Claude",
                                "chrome": "Google Chrome",
                                "crome": "Google Chrome",
                                "krome": "Google Chrome",
                            }
                            app_lower = app_name.lower()
                            if app_lower in app_corrections:
                                corrected = app_corrections[app_lower]
                                logger.info(f"[Action] Correcting '{app_name}' -> '{corrected}'")
                                app_name = corrected

                            logger.info(f"[Action] Opening app: {app_name}")
                            try:
                                result = self.agent.control.open_app(app_name)
                                logger.info(f"[Action] open_app returned: {result}")
                                if result:
                                    logger.info(f"[Action] Successfully opened {app_name}")
                                    return json.dumps({"success": True, "message": f"Opened {app_name}"})
                                else:
                                    logger.info(f"[Action] Failed to open {app_name} - app not found?")
                                    return json.dumps({"success": False, "error": f"Could not open '{app_name}'. App may not exist."})
                            except Exception as open_err:
                                logger.error(f"[Action] open_app failed: {open_err}")
                                return json.dumps({"success": False, "error": str(open_err)})

                        elif name == "open_url":
                            url = args.get("url", "")
                            # Add https:// if no protocol specified
                            if url and not url.startswith(("http://", "https://")):
                                url = "https://" + url
                            success = self.agent.control.open_url(url)
                            if success:
                                return json.dumps({"success": True, "url": url})
                            else:
                                return json.dumps({"success": False, "error": f"Failed to open {url}"})

                        elif name == "scroll":
                            logger.info(f"[DEBUG] MATCHED: scroll")
                            amount = args.get("amount", 0)
                            logger.info(f"[Action] Scroll: {amount}")
                            result = self.agent.control.scroll(amount)
                            logger.info(f"[Action] Scroll result: {result}")
                            return json.dumps({"success": result, "amount": amount})

                        elif name == "web_search":
                            query = args.get("query", "")
                            print(f"[Action] Web search: {query}")
                            try:
                                # Use Gemini with Google Search grounding (works even with OpenAI voice)
                                from google import genai
                                from google.genai import types as genai_types
                                search_client = genai.Client(api_key=GOOGLE_API_KEY)

                                response = search_client.models.generate_content(
                                    model="gemini-2.0-flash-exp",
                                    contents=f"Search the web and provide a concise answer to: {query}",
                                    config=genai_types.GenerateContentConfig(
                                        tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())]
                                    )
                                )

                                result_text = response.text if response.text else "No results found"
                                print(f"[Action] Web search result: {result_text[:200]}...")
                                return json.dumps({"success": True, "query": query, "result": result_text})
                            except Exception as search_err:
                                print(f"[Action] Web search error: {search_err}")
                                return json.dumps({"success": False, "error": str(search_err)})

                        elif name == "double_click":
                            target = args.get("target", "")
                            print(f"[Action] Double-click: {target}")
                            element = action_executor.find_element(target)
                            if element:
                                x, y = element["x"], element["y"]
                                print(f"[Action] Found at ({x}, {y}), double-clicking")
                                result = self.agent.control.double_click(x, y)
                                return json.dumps({"success": result, "target": target, "x": x, "y": y})
                            return json.dumps({"success": False, "error": f"Could not find: {target}"})

                        elif name == "right_click":
                            target = args.get("target", "")
                            print(f"[Action] Right-click: {target}")
                            element = action_executor.find_element(target)
                            if element:
                                x, y = element["x"], element["y"]
                                print(f"[Action] Found at ({x}, {y}), right-clicking")
                                result = self.agent.control.right_click(x, y)
                                return json.dumps({"success": result, "target": target, "x": x, "y": y})
                            return json.dumps({"success": False, "error": f"Could not find: {target}"})

                        elif name == "move_mouse":
                            logger.info(f"[DEBUG] MATCHED: move_mouse")
                            target = args.get("target", "")
                            logger.info(f"[Action] Move mouse to: {target}")
                            element = action_executor.find_element(target)
                            logger.info(f"[Action] find_element returned: {element}")
                            if element:
                                x, y = element["x"], element["y"]
                                logger.info(f"[Action] Moving mouse to ({x}, {y})")
                                result = self.agent.control.move_to(x, y)
                                logger.info(f"[Action] move_to result: {result}")
                                return json.dumps({"success": result, "target": target, "x": x, "y": y})
                            logger.info(f"[Action] Could not find element: {target}")
                            return json.dumps({"success": False, "error": f"Could not find: {target}"})

                        elif name == "drag":
                            from_target = args.get("from_target", "")
                            to_target = args.get("to_target", "")
                            print(f"[Action] Drag from '{from_target}' to '{to_target}'")
                            from_element = action_executor.find_element(from_target)
                            to_element = action_executor.find_element(to_target)
                            if from_element and to_element:
                                start_x, start_y = from_element["x"], from_element["y"]
                                end_x, end_y = to_element["x"], to_element["y"]
                                print(f"[Action] Dragging from ({start_x}, {start_y}) to ({end_x}, {end_y})")
                                result = self.agent.control.drag_to(start_x, start_y, end_x, end_y)
                                return json.dumps({"success": result, "from": [start_x, start_y], "to": [end_x, end_y]})
                            error = f"Could not find: {from_target if not from_element else to_target}"
                            return json.dumps({"success": False, "error": error})

                        elif name == "get_mouse_position":
                            import pyautogui
                            x, y = pyautogui.position()
                            print(f"[Action] Mouse position: ({x}, {y})")
                            return json.dumps({"x": x, "y": y, "screen_size": [self.agent.control.screen_width, self.agent.control.screen_height]})

                        elif name == "click_at_coordinates":
                            x = args.get("x", 0)
                            y = args.get("y", 0)
                            button = args.get("button", "left")

                            # Scale coordinates from screenshot space to actual screen space
                            # Gemini sees screenshots at SCREENSHOT_MAX_WIDTH, so ALWAYS scale if screen is larger
                            from .config import SCREENSHOT_MAX_WIDTH
                            screen_width = self.agent.control.screen_width
                            screen_height = self.agent.control.screen_height

                            if screen_width > SCREENSHOT_MAX_WIDTH:
                                scale = screen_width / SCREENSHOT_MAX_WIDTH
                                scaled_x = int(x * scale)
                                scaled_y = int(y * scale)
                                logger.info(f"[Action] Click at ({x}, {y}) -> scaled to ({scaled_x}, {scaled_y}) [scale: {scale:.2f}x] with {button} button")
                            else:
                                scaled_x = int(x)
                                scaled_y = int(y)
                                logger.info(f"[Action] Click at ({x}, {y}) (no scaling needed) with {button} button")
                            result = self.agent.control.click(scaled_x, scaled_y, button=button)
                            return json.dumps({"success": result, "x": scaled_x, "y": scaled_y, "button": button, "original": [x, y]})

                        else:
                            logger.info(f"[DEBUG] UNMATCHED tool: {name}")
                            return f'{{"error": "Unknown tool: {name}"}}'

                    except Exception as e:
                        logger.error(f"[DEBUG] EXCEPTION in tool handler: {e}")
                        import traceback
                        traceback.print_exc()
                        return f'{{"error": "{str(e)}"}}'

                # Create conversation loop
                loop = RealtimeConversationLoop(
                    api_key=OPENAI_API_KEY,
                    config=config,
                    tools=ARIA_REALTIME_TOOLS,
                    tool_handler=handle_tool
                )

                # ===== AUTOMATIC LEARNING SYSTEM =====
                # Track each conversation turn for automatic knowledge extraction
                turn_state = {
                    "user_input": "",
                    "tool_calls": [],
                }

                # Wrap the tool handler to track tool calls AND record outcomes
                original_handle_tool = handle_tool
                def tracking_handle_tool(call_id: str, name: str, args: dict) -> str:
                    result = original_handle_tool(call_id, name, args)

                    # Track for learning
                    turn_state["tool_calls"].append({
                        "name": name,
                        "args": args,
                        "result": result[:500] if len(result) > 500 else result
                    })

                    # Record action outcome
                    try:
                        result_data = json.loads(result)
                        success = result_data.get("success", True)
                        if "error" in result_data:
                            success = False

                        def record():
                            try:
                                context = f"user_goal: {turn_state.get('user_input', '')[:100]}"
                                self.agent.memory.record_action_outcome(name, args, success, context)
                            except Exception:
                                pass
                        threading.Thread(target=record, daemon=True).start()
                    except (json.JSONDecodeError, AttributeError):
                        pass

                    return result

                loop.tool_handler = tracking_handle_tool

                # Set up callbacks with automatic learning
                def on_user_speech(text):
                    logger.info(f"[User]: {text}")
                    self._update_status("Thinking...")
                    turn_state["user_input"] = text
                    turn_state["tool_calls"] = []

                def on_assistant_done(text):
                    logger.info(f"[Aria]: {text[:100]}...")
                    self._update_status("Listening...")

                    # Automatic learning
                    user_input = turn_state.get("user_input", "")
                    tool_calls = turn_state.get("tool_calls", [])

                    if user_input and text:
                        def extract_learnings():
                            try:
                                logger.info(f"[Learning] Extracting knowledge from turn...")
                                self.agent.memory.extract_and_store_memories(
                                    user_input=user_input,
                                    assistant_response=text,
                                    actions_taken=tool_calls
                                )
                                logger.info(f"[Learning] Knowledge extraction complete")
                            except Exception as learn_err:
                                logger.warning(f"[Learning] Failed to extract: {learn_err}")

                        threading.Thread(target=extract_learnings, daemon=True).start()

                    turn_state["user_input"] = ""
                    turn_state["tool_calls"] = []

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

    def _needs_computer_control(self, user_input: str) -> bool:
        """
        HYBRID ARCHITECTURE ROUTING: Determine if task needs Claude Computer Use.

        This is a critical routing decision in the hybrid architecture:

        Returns True -> Route to Claude Computer Use agent
            - Tasks requiring mouse, keyboard, or screen interaction
            - File/folder operations, app control, browser navigation
            - Anything that needs to "do something" on the computer

        Returns False -> Can use faster conversational response
            - Simple Q&A, chitchat, knowledge questions
            - Memory operations (remember/recall)
            - Tasks that don't require computer interaction

        The goal is to use Claude's superior reasoning for complex computer tasks
        while keeping simple interactions fast via the Realtime API.
        """
        user_lower = user_input.lower()

        # Keywords that indicate computer control is needed
        COMPUTER_KEYWORDS = [
            "click", "type", "scroll", "mouse", "keyboard",
            "window", "tab", "browser", "file", "folder",
            "desktop", "dock", "menu", "select", "copy", "paste",
            "drag", "move", "maximize", "minimize", "close", "resize",
            "screenshot", "screen", "what's on", "what do you see",
        ]

        # Action verbs that suggest computer control
        ACTION_VERBS = [
            "open", "go to", "navigate", "find", "search for",
            "show me", "take me", "switch to", "close", "quit",
        ]

        if any(keyword in user_lower for keyword in COMPUTER_KEYWORDS):
            return True
        if any(verb in user_lower for verb in ACTION_VERBS):
            return True
        return False

    def _run_claude_task(self, task: str) -> str:
        """
        Run a task using Claude Computer Use agent.
        Returns the result to speak back to user.
        """
        if not self.claude_agent:
            return "Sorry, Claude Computer Use is not available."

        logger.info(f"[Claude Task]: {task}")
        self._update_status("Claude working...")

        try:
            result = self.claude_agent.run(task)
            # Truncate for voice
            if result and len(result) > 300:
                result = result[:300] + "..."
            return result or "Done."
        except Exception as e:
            logger.error(f"Claude task error: {e}")
            return f"Sorry, I encountered an error: {str(e)[:50]}"

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
