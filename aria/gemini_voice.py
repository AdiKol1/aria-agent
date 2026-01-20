"""
Gemini 2.0 Flash Live Voice for Aria.

Provides real-time voice-to-voice using Google's Gemini Live API.
This provides superior reasoning capabilities compared to OpenAI's Realtime API
while maintaining low latency (~300ms) for voice conversations.
"""

import asyncio
import base64
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pyaudio

from .config import DISABLE_FALLBACK_SYSTEM, DISABLE_IMMEDIATE_FALLBACK, DISABLE_CONFABULATION_FALLBACK

# Import the new VoiceBridge for reliable action execution
try:
    from .core import VoiceBridge, get_voice_bridge
    VOICE_BRIDGE_AVAILABLE = True
except ImportError:
    VOICE_BRIDGE_AVAILABLE = False
    VoiceBridge = None
    get_voice_bridge = None

logger = logging.getLogger("aria")

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None


@dataclass
class GeminiVoiceConfig:
    """Configuration for Gemini Live Voice API."""
    model: str = "gemini-2.0-flash-exp"
    voice: str = "Puck"  # Options: Puck, Charon, Kore, Fenrir, Aoede
    sample_rate: int = 16000  # Gemini Live requires 16kHz
    output_sample_rate: int = 24000
    instructions: str = ""


class GeminiVoiceClient:
    """Gemini Live API client for voice interaction."""

    def __init__(self, api_key: str, config: Optional[GeminiVoiceConfig] = None):
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-genai package is required. Install with: pip install google-genai"
            )

        self.api_key = api_key
        self.config = config or GeminiVoiceConfig()

        # Initialize the Gemini client
        self.client = genai.Client(api_key=api_key)
        self.session = None
        self._active_session = None

        # Audio handling - use thread-safe queue
        self._audio: Optional[pyaudio.PyAudio] = None
        self._input_stream: Optional[pyaudio.Stream] = None
        self._output_stream: Optional[pyaudio.Stream] = None
        self._audio_output_queue: queue.Queue = queue.Queue()
        self._audio_input_queue: queue.Queue = queue.Queue()  # Thread-safe queue

        # Screen capture for vision
        self._screen_queue: queue.Queue = queue.Queue()
        self._screen_capture_enabled = True  # Enable vision for Gemini
        self._screen_capture_interval = 2.0  # Send screen every 2 seconds
        self._last_screen_time = 0

        # State
        self.is_connected = False
        self.is_speaking = False
        self.is_listening = False
        self._stop_audio = threading.Event()
        self._output_thread: Optional[threading.Thread] = None
        self._speech_ended_time: float = 0
        self._event_loop = None
        self._accumulated_text = ""

        # Tool handling
        self._tools: List[Dict[str, Any]] = []

        # Fallback execution - track if tools were actually called
        self._tool_called_this_turn: bool = False
        self._last_user_request: str = ""  # What the user asked for (accumulated)
        self._last_action_request: str = ""  # The ORIGINAL action request (preserved for fallback)
        self._current_user_transcript: str = ""  # Accumulating transcript for current turn
        self._confabulation_correction_count: int = 0  # Limit corrections to prevent loops
        self._max_confabulation_corrections: int = 3  # Max retries before giving up (includes fallback execution)
        self._last_transcript_time: float = 0  # When we last received transcript (for accumulation)

        # Rolling transcript buffer - keeps recent speech fragments for better context
        self._transcript_buffer: list = []  # List of (timestamp, text) tuples
        self._transcript_buffer_window: float = 10.0  # Keep last 10 seconds of speech
        self._immediate_fallback_triggered: bool = False  # Track if we already triggered fallback for this request

        # Action deduplication - prevent executing same action multiple times
        self._recent_actions: dict = {}  # {action_key: timestamp} - tracks recent actions
        self._action_cooldown: float = 5.0  # Don't repeat same action within 5 seconds

        # Anti-repetition - prevent saying the same thing multiple times
        self._recent_responses: list = []  # Last few responses
        self._max_recent_responses: int = 5  # Track last 5 responses
        self._fallback_completed: bool = False  # Flag to indicate fallback already handled request

        # Callbacks
        self.on_transcript: Optional[Callable[[str], None]] = None
        self.on_response_text: Optional[Callable[[str], None]] = None
        self.on_response_done: Optional[Callable[[str], None]] = None
        self.on_tool_call: Optional[Callable[[str, str, Dict], str]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_speech_started: Optional[Callable[[], None]] = None
        self.on_speech_stopped: Optional[Callable[[], None]] = None

        # VoiceBridge for reliable action execution (bypasses Gemini tool calling)
        self._voice_bridge: Optional[Any] = None
        if VOICE_BRIDGE_AVAILABLE:
            try:
                self._voice_bridge = get_voice_bridge(use_ai_fallback=True)
                logger.info("[GeminiVoice] VoiceBridge initialized for direct action execution")
            except Exception as e:
                logger.warning(f"[GeminiVoice] Failed to initialize VoiceBridge: {e}")

    def add_tools(self, tools: List[Dict[str, Any]]) -> None:
        """Add function calling tools to the session."""
        self._tools = tools
        print(f"[GeminiVoice] Registered {len(self._tools)} tools")

    def _try_voice_bridge(self, transcript: str) -> bool:
        """Try to process transcript through VoiceBridge for immediate action execution.

        This bypasses Gemini's unreliable tool calling by parsing and executing
        actions directly through the AriaEngine.

        Args:
            transcript: The user's transcribed speech.

        Returns:
            True if VoiceBridge handled the request, False otherwise.
        """
        if not self._voice_bridge:
            logger.info(f"[VoiceBridge] No bridge available")
            return False

        try:
            logger.info(f"[VoiceBridge] Checking transcript: '{transcript[:50]}...' (len={len(transcript)})")
            # Check if this looks like an action command
            if not self._voice_bridge.should_intercept(transcript):
                logger.info(f"[VoiceBridge] Not intercepting: '{transcript[:50]}...'")
                return False

            logger.info(f"[VoiceBridge] Intercepting action request: '{transcript}'")

            # PRE-EXECUTION DEDUP: Parse intent to see what action WOULD be executed
            # Then check if that action was recently executed
            try:
                from aria.core.intent_parser import parse
                intent = parse(transcript)
                # Create a dedup key from the parsed intent (action + target)
                pre_check_key = f"{intent.action.value}:{intent.target or ''}".lower().strip()
                now = time.time()

                if pre_check_key in self._recent_actions:
                    time_since = now - self._recent_actions[pre_check_key]
                    if time_since < self._action_cooldown:
                        logger.info(f"[VoiceBridge] SKIPPING duplicate intent (executed {time_since:.1f}s ago): '{pre_check_key}'")
                        return True  # Block without executing
            except Exception as e:
                logger.debug(f"[VoiceBridge] Pre-dedup check failed: {e}")
                pre_check_key = None

            # Process through VoiceBridge (deterministic, no AI needed for common commands)
            result = self._voice_bridge.process_voice_input(transcript)

            if result.success:
                logger.info(f"[VoiceBridge] Action executed: {result.response}")

                # Record this action for deduplication using intent key
                if pre_check_key:
                    self._recent_actions[pre_check_key] = time.time()
                    logger.debug(f"[VoiceBridge] Recorded action for dedup: '{pre_check_key}'")

                    # ALSO record in tool handler format for cross-system deduplication
                    # Map intent to tool call format: open_app:{"app": "Google Chrome"}
                    try:
                        from aria.core.intent_parser import IntentType
                        import json
                        tool_key = None
                        if intent.action == IntentType.OPEN and intent.target:
                            # Map common apps
                            target_lower = intent.target.lower()
                            app_map = {
                                "chrome": "Google Chrome", "google chrome": "Google Chrome",
                                "safari": "Safari", "finder": "Finder", "terminal": "Terminal",
                                "notes": "Notes", "messages": "Messages", "slack": "Slack",
                                "claude": "Claude", "vscode": "Visual Studio Code", "code": "Visual Studio Code",
                            }
                            app_name = app_map.get(target_lower, intent.target.title())
                            tool_key = f'open_app:{json.dumps({"app": app_name}, sort_keys=True)}'
                        elif intent.action == IntentType.NAVIGATE and intent.target:
                            url = intent.target if '://' in intent.target else f'https://{intent.target}'
                            tool_key = f'open_url:{json.dumps({"url": url}, sort_keys=True)}'
                        elif intent.action == IntentType.SCROLL:
                            amount = -300 if 'down' in (intent.target or '').lower() else 300
                            tool_key = f'scroll:{json.dumps({"amount": amount}, sort_keys=True)}'

                        if tool_key:
                            self._recent_actions[tool_key] = time.time()
                            logger.debug(f"[VoiceBridge] Also recorded tool-format dedup key: '{tool_key}'")
                    except Exception as e:
                        logger.debug(f"[VoiceBridge] Could not create tool-format dedup key: {e}")

                # Mark that we handled this turn - prevents confabulation fallback
                self._tool_called_this_turn = True
                self._last_action_request = ""  # Clear since we handled it

                # Queue a spoken response if we have TTS capability
                # The response will be spoken after Gemini's turn ends
                if result.response and hasattr(self, '_speak_response'):
                    asyncio.create_task(self._speak_response(result.response))

                return True
            else:
                logger.warning(f"[VoiceBridge] Action failed: {result.error}")
                # Let Gemini handle it - maybe it can do better
                return False

        except Exception as e:
            logger.error(f"[VoiceBridge] Error processing: {e}")
            return False

    async def _speak_response(self, text: str) -> None:
        """Speak a response using TTS (if available)."""
        try:
            # Try to use the on_response_done callback to trigger speech
            if self.on_response_done:
                self.on_response_done(f"Done. {text}")
        except Exception as e:
            logger.debug(f"[VoiceBridge] Could not speak response: {e}")

    async def connect(self) -> bool:
        """Establish connection to Gemini Live API."""
        try:
            # Build function declarations from tools
            function_declarations = None
            if self._tools:
                function_declarations = [
                    types.FunctionDeclaration(
                        name=tool["name"],
                        description=tool.get("description", ""),
                        parameters=tool.get("parameters")
                    )
                    for tool in self._tools
                ]

            # Build the live config using proper types
            # Note: Using AUDIO only - combining AUDIO+TEXT with voice/tools/system_instruction fails
            # Note: tool_config is NOT supported by LiveConnectConfig
            live_config = types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.config.voice
                        )
                    )
                ),
                # Enable transcription of user's speech input
                input_audio_transcription=types.AudioTranscriptionConfig(),
                # Enable transcription of model's audio output
                output_audio_transcription=types.AudioTranscriptionConfig(),
                system_instruction=self.config.instructions if self.config.instructions else None,
                tools=[types.Tool(function_declarations=function_declarations)] if function_declarations else None
            )

            # Connect to the Live API
            self.session = self.client.aio.live.connect(
                model=self.config.model,
                config=live_config
            )

            self.is_connected = True
            print("[GeminiVoice] Connected to Gemini Live API")
            return True

        except Exception as e:
            print(f"[GeminiVoice] Failed to connect: {e}")
            import traceback
            traceback.print_exc()
            if self.on_error:
                self.on_error(f"Connection failed: {e}")
            return False

    async def run_session(self) -> None:
        """Run the main session loop."""
        try:
            print("[GeminiVoice] Starting session...")
            async with self.session as session:
                self._active_session = session
                print("[GeminiVoice] Session context entered")

                # NOTE: Initial text workaround disabled - was causing issues
                # See: https://github.com/googleapis/python-genai/issues/843
                print("[GeminiVoice] Session ready, starting audio/screen tasks...")

                # Start tasks - audio, screen, and receive
                send_audio_task = asyncio.create_task(self._send_audio_loop(session))
                send_screen_task = asyncio.create_task(self._send_screen_loop(session))
                receive_task = asyncio.create_task(self._receive_loop(session))

                # Wait for any to complete (usually due to error/disconnect)
                done, pending = await asyncio.wait(
                    [send_audio_task, send_screen_task, receive_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Log which task completed and check for errors
                task_error = None
                for task in done:
                    if task == send_audio_task:
                        print("[GeminiVoice] Audio send task completed first")
                    elif task == send_screen_task:
                        print("[GeminiVoice] Screen send task completed first")
                    elif task == receive_task:
                        print("[GeminiVoice] Receive task completed first")

                    # Check if task raised an exception
                    if task.exception() is not None:
                        task_error = task.exception()
                        print(f"[GeminiVoice] Task error: {task_error}")

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Re-raise task error to trigger reconnection
                if task_error:
                    raise task_error

            print("[GeminiVoice] Session context exited")

        except Exception as e:
            print(f"[GeminiVoice] Session error: {e}")
            import traceback
            traceback.print_exc()
            self.is_connected = False
            self._active_session = None
            if self.on_error:
                self.on_error(str(e))
            # Re-raise to trigger reconnection in GeminiConversationLoop
            raise
        finally:
            # Always mark as disconnected when session ends
            self.is_connected = False
            self._active_session = None

    async def _send_audio_loop(self, session) -> None:
        """Send audio from queue to Gemini."""
        while self.is_connected:
            try:
                # Check queue (non-blocking)
                try:
                    audio_data = self._audio_input_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.02)  # 20ms
                    continue

                # Send audio to Gemini using proper types
                # Audio should be raw bytes, not base64
                await session.send(
                    input=types.LiveClientRealtimeInput(
                        mediaChunks=[
                            types.Blob(
                                data=audio_data,
                                mimeType="audio/pcm;rate=16000"
                            )
                        ]
                    ),
                    end_of_turn=False
                )

            except Exception as e:
                if self.is_connected:
                    print(f"[GeminiVoice] Error sending audio: {e}")
                await asyncio.sleep(0.1)

    async def _send_screen_loop(self, session) -> None:
        """Periodically send screen captures to Gemini for continuous vision."""
        from .vision import get_screen_capture

        screen_capture = get_screen_capture()
        print("[GeminiVoice] Screen capture loop started")

        while self.is_connected:
            try:
                now = time.time()

                # Check if it's time to send a screen update
                if self._screen_capture_enabled and (now - self._last_screen_time) >= self._screen_capture_interval:
                    # Don't send screen while Aria is speaking (avoid feedback)
                    if not self.is_speaking:
                        # Capture screen as JPEG bytes
                        result = screen_capture.capture_to_base64_with_size()
                        if result:
                            image_b64, (width, height) = result
                            # Decode base64 to raw bytes
                            image_bytes = base64.b64decode(image_b64)

                            # Send image to Gemini using same API pattern as audio
                            await session.send(
                                input=types.LiveClientRealtimeInput(
                                    mediaChunks=[
                                        types.Blob(
                                            data=image_bytes,
                                            mimeType="image/jpeg"
                                        )
                                    ]
                                ),
                                end_of_turn=False
                            )
                            self._last_screen_time = now
                            print(f"[GeminiVoice] Sent screen capture ({width}x{height})")

                await asyncio.sleep(0.5)  # Check every 500ms

            except Exception as e:
                if self.is_connected:
                    print(f"[GeminiVoice] Error sending screen: {e}")
                    import traceback
                    traceback.print_exc()
                await asyncio.sleep(1.0)

    async def send_screen_now(self) -> bool:
        """Send an immediate screen capture to Gemini."""
        if not self._active_session:
            return False

        try:
            from .vision import get_screen_capture
            screen_capture = get_screen_capture()
            result = screen_capture.capture_to_base64_with_size()

            if result:
                image_b64, (width, height) = result
                image_bytes = base64.b64decode(image_b64)

                # Send image using same API pattern as audio
                await self._active_session.send(
                    input=types.LiveClientRealtimeInput(
                        mediaChunks=[
                            types.Blob(
                                data=image_bytes,
                                mimeType="image/jpeg"
                            )
                        ]
                    ),
                    end_of_turn=False
                )
                self._last_screen_time = time.time()
                print(f"[GeminiVoice] Sent immediate screen capture ({width}x{height})")
                return True
        except Exception as e:
            print(f"[GeminiVoice] Error sending immediate screen: {e}")
            import traceback
            traceback.print_exc()
        return False

    async def _receive_loop(self, session) -> None:
        """Process responses from Gemini.

        IMPORTANT: receive() exits when a turn completes, so we wrap it in
        a while loop to continue receiving across multiple turns.
        See: https://github.com/googleapis/python-genai/issues/1224
        """
        self._accumulated_text = ""
        turn_count = 0

        try:
            print("[GeminiVoice] Starting receive loop...")
            # Wrap in while loop - receive() exits after each turn_complete
            while self.is_connected:
                turn_count += 1
                print(f"[GeminiVoice] Listening for turn {turn_count}...")
                async for response in session.receive():
                    await self._handle_response(response)
                # Turn completed, loop continues for next turn
                print(f"[GeminiVoice] Turn {turn_count} complete, ready for next turn")

        except asyncio.CancelledError:
            print(f"[GeminiVoice] Receive loop cancelled after {turn_count} turns")
            raise
        except Exception as e:
            error_str = str(e)
            print(f"[GeminiVoice] Error in receive loop at turn {turn_count}: {e}")
            import traceback
            traceback.print_exc()

            # Check for WebSocket/connection errors that should trigger reconnection
            is_connection_error = any(err in error_str.lower() for err in [
                "1011",  # Service unavailable
                "1006",  # Abnormal closure
                "connection",
                "websocket",
                "unavailable",
                "eof",
            ])

            if is_connection_error:
                print(f"[GeminiVoice] Connection error detected, signaling for reconnection")
                self.is_connected = False
                if self.on_error:
                    self.on_error(f"Connection lost: {error_str}")
                # Re-raise to trigger reconnection in outer loop
                raise

    async def _handle_response(self, response) -> None:
        """Handle incoming response from Gemini."""
        try:
            # Debug: Log all response attributes to understand what Gemini sends
            resp_type = type(response).__name__
            resp_attrs = [a for a in dir(response) if not a.startswith('_')]

            # Check for different transcription locations
            # Gemini might send transcription in various ways
            if hasattr(response, 'text') and response.text:
                print(f"[DEBUG] Response has text: {response.text[:100]}")

            if hasattr(response, 'transcript') and response.transcript:
                print(f"[User said]: {response.transcript}")
                now = time.time()

                # Add to rolling buffer and clean old entries
                self._transcript_buffer.append((now, response.transcript))
                self._transcript_buffer = [(t, txt) for t, txt in self._transcript_buffer
                                           if now - t <= self._transcript_buffer_window]

                # Build full request from rolling buffer (last 10 seconds of speech)
                full_request = " ".join(txt for _, txt in self._transcript_buffer)
                self._last_user_request = full_request.strip()

                # Check if this is a new request (gap > 5 seconds)
                if now - self._last_transcript_time > 5.0:
                    self._tool_called_this_turn = False
                    self._confabulation_correction_count = 0
                    self._immediate_fallback_triggered = False  # Reset for new request
                    self._current_user_transcript = response.transcript
                else:
                    self._current_user_transcript += " " + response.transcript

                self._last_transcript_time = now

                # ============================================================
                # VOICEBRIDGE INTEGRATION: Try to handle action immediately
                # This bypasses Gemini's unreliable tool calling for common commands
                # ============================================================
                if self._voice_bridge and not self._tool_called_this_turn:
                    # Try VoiceBridge first - it handles common commands deterministically
                    if self._try_voice_bridge(self._last_user_request):
                        # VoiceBridge handled it - skip fallback logic
                        logger.info(f"[VoiceBridge] Successfully handled: '{self._last_user_request}'")
                        # CRITICAL: Clear transcript buffer to prevent re-triggering
                        # When user says "Yes" after "Open Chrome", buffer becomes "Open Chrome. Yes"
                        # which would trigger another action. Clearing prevents this.
                        self._transcript_buffer = []
                        self._last_user_request = ""
                        logger.info(f"[VoiceBridge] Cleared transcript buffer to prevent re-triggering")
                        if self.on_transcript:
                            self.on_transcript(response.transcript)
                        # Continue to next response - Gemini may still respond but we've done the action
                        # The _tool_called_this_turn flag prevents confabulation fallback

                # ALWAYS check for action keywords and store the action request
                # This is used for fallback when Gemini confabulates (if VoiceBridge didn't handle it)
                action_keywords = ['open', 'click', 'scroll', 'type', 'press', 'close', 'newtab', 'search',
                                   'move', 'mouse', 'goto', 'navigate', 'drag', 'double', 'copy', 'paste',
                                   'undo', 'redo', 'select', 'delete', 'clear', 'write', 'enter', 'send']
                request_lower = self._last_user_request.lower()
                request_normalized = request_lower.replace(' ', '')

                # If this looks like an action request, store it for fallback use
                if any(keyword in request_lower or keyword in request_normalized for keyword in action_keywords):
                    if not self._tool_called_this_turn:  # Only store if VoiceBridge didn't handle it
                        self._last_action_request = self._last_user_request
                        logger.info(f"[ACTION REQUEST] Stored action request: '{self._last_action_request}'")

                # IMMEDIATE FALLBACK: Trigger fallback RIGHT NOW before Gemini speaks
                # NOTE: DISABLED - VoiceBridge now handles this more reliably
                if not DISABLE_IMMEDIATE_FALLBACK and not self._tool_called_this_turn and not self._immediate_fallback_triggered:
                    if any(keyword in request_lower or keyword in request_normalized for keyword in action_keywords):
                        logger.info(f"[IMMEDIATE FALLBACK] Detected action request: '{self._last_user_request}'")
                        self._immediate_fallback_triggered = True
                        # Small delay to let more speech accumulate, then trigger fallback
                        asyncio.create_task(self._delayed_fallback(0.2))

                if self.on_transcript:
                    self.on_transcript(response.transcript)

            # Check for user input/speech in server_content
            if hasattr(response, 'server_content') and response.server_content:
                content = response.server_content

                # Log content attributes for debugging
                content_attrs = [a for a in dir(content) if not a.startswith('_') and not callable(getattr(content, a, None))]

                # Check various possible transcription field names
                for field in ['input_transcription', 'transcription', 'user_input', 'speech_transcription', 'input_text']:
                    if hasattr(content, field):
                        field_val = getattr(content, field)
                        if field_val:
                            transcript_text = None
                            if hasattr(field_val, 'text') and field_val.text:
                                transcript_text = field_val.text
                            elif isinstance(field_val, str) and field_val:
                                transcript_text = field_val

                            if transcript_text:
                                print(f"[User said]: {transcript_text}")
                                now = time.time()

                                # Add to rolling buffer
                                self._transcript_buffer.append((now, transcript_text))
                                self._transcript_buffer = [(t, txt) for t, txt in self._transcript_buffer
                                                           if now - t <= self._transcript_buffer_window]

                                # Build full request from rolling buffer
                                full_request = " ".join(txt for _, txt in self._transcript_buffer)
                                self._last_user_request = full_request.strip()

                                # Check if this is a new request
                                if now - self._last_transcript_time > 5.0:
                                    self._tool_called_this_turn = False
                                    self._confabulation_correction_count = 0
                                    self._immediate_fallback_triggered = False
                                    self._current_user_transcript = transcript_text
                                else:
                                    self._current_user_transcript += " " + transcript_text

                                self._last_transcript_time = now

                                # ============================================================
                                # VOICEBRIDGE INTEGRATION: Try to handle action immediately
                                # ============================================================
                                if self._voice_bridge and not self._tool_called_this_turn:
                                    if self._try_voice_bridge(self._last_user_request):
                                        logger.info(f"[VoiceBridge] Successfully handled: '{self._last_user_request}'")
                                        # CRITICAL: Clear transcript buffer to prevent re-triggering
                                        self._transcript_buffer = []
                                        self._last_user_request = ""
                                        logger.info(f"[VoiceBridge] Cleared transcript buffer to prevent re-triggering")
                                        if self.on_transcript:
                                            self.on_transcript(transcript_text)
                                        continue  # Skip to next field - action already handled

                                # ALWAYS check for action keywords and store the action request
                                action_keywords = ['open', 'click', 'scroll', 'type', 'press', 'close', 'newtab', 'search',
                                                   'move', 'mouse', 'goto', 'navigate', 'drag', 'double', 'copy', 'paste',
                                                   'undo', 'redo', 'select', 'delete', 'clear', 'write', 'enter', 'send']
                                request_lower = self._last_user_request.lower()
                                request_normalized = request_lower.replace(' ', '')

                                # If this looks like an action request, store it for fallback use
                                if any(keyword in request_lower or keyword in request_normalized for keyword in action_keywords):
                                    if not self._tool_called_this_turn:  # Only store if VoiceBridge didn't handle it
                                        self._last_action_request = self._last_user_request
                                        logger.info(f"[ACTION REQUEST] Stored action request: '{self._last_action_request}'")

                                # IMMEDIATE FALLBACK check
                                # NOTE: DISABLED - VoiceBridge now handles this more reliably
                                if not DISABLE_IMMEDIATE_FALLBACK and not self._tool_called_this_turn and not self._immediate_fallback_triggered:
                                    if any(keyword in request_lower or keyword in request_normalized for keyword in action_keywords):
                                        logger.info(f"[IMMEDIATE FALLBACK] Detected action in transcript: '{self._last_user_request}'")
                                        self._immediate_fallback_triggered = True
                                        asyncio.create_task(self._delayed_fallback(0.2))

                                if self.on_transcript:
                                    self.on_transcript(transcript_text)

            # Handle different response types
            if hasattr(response, 'server_content') and response.server_content:
                content = response.server_content

                # Handle interruption (user barged in)
                if hasattr(content, 'interrupted') and content.interrupted:
                    print("[GeminiVoice] Interrupted by user - clearing audio queue")
                    # Clear audio output queue to stop playback
                    while not self._audio_output_queue.empty():
                        try:
                            self._audio_output_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.is_speaking = False
                    return

                # Check for OUTPUT transcription (what Aria says) - this is separate from input transcription
                if hasattr(content, 'output_transcription') and content.output_transcription:
                    output_trans = content.output_transcription
                    if hasattr(output_trans, 'text') and output_trans.text:
                        aria_text = output_trans.text
                        print(f"[Aria says]: {aria_text}")
                        self._accumulated_text += aria_text

                        # CRITICAL FIX: Detect if Gemini is SAYING function calls instead of CALLING them
                        text_lower = aria_text.lower()
                        function_call_patterns = ['move_mouse(', 'click(', 'scroll(', 'open_app(', 'type_text(',
                                                  'press_key(', 'hotkey(', 'fill_field(', 'open_url(']
                        if any(pattern in text_lower for pattern in function_call_patterns):
                            logger.warning(f"[CONFABULATION] Gemini SPOKE function call: '{aria_text}'")
                            # Clear audio queue to stop speaking
                            while not self._audio_output_queue.empty():
                                try:
                                    self._audio_output_queue.get_nowait()
                                except:
                                    break
                            # Parse and execute the function call
                            asyncio.create_task(self._parse_and_execute_spoken_function(aria_text))

                        if self.on_response_text:
                            self.on_response_text(aria_text)

                # Handle model output
                if hasattr(content, 'model_turn') and content.model_turn:
                    for part in content.model_turn.parts:
                        # Audio output
                        if hasattr(part, 'inline_data') and part.inline_data:
                            audio_data = part.inline_data.data
                            if audio_data:
                                # Decode if base64
                                if isinstance(audio_data, str):
                                    audio_data = base64.b64decode(audio_data)
                                audio_len = len(audio_data) if audio_data else 0
                                # Don't spam audio chunk logs
                                self._audio_output_queue.put(audio_data)
                                if not self.is_speaking:
                                    self.is_speaking = True
                                    print(f"[GeminiVoice] Aria speaking...")

                                    # EARLY FALLBACK: If Aria starts speaking without calling a tool for an action request,
                                    # trigger fallback NOW before she finishes speaking something useless
                                    # NOTE: DISABLED - Causes garbage actions, also fires too early
                                    if not DISABLE_IMMEDIATE_FALLBACK and self._last_user_request and not self._tool_called_this_turn:
                                        user_request_lower = self._last_user_request.lower()
                                        user_request_normalized = user_request_lower.replace(' ', '')
                                        action_keywords = ['open', 'click', 'type', 'fill', 'scroll', 'search', 'close', 'press',
                                                           'write', 'enter', 'delete', 'remove', 'send', 'message',
                                                           'move', 'mouse', 'goto', 'navigate', 'drag', 'double']
                                        user_requested_action = any(keyword in user_request_lower or keyword in user_request_normalized for keyword in action_keywords)
                                        if user_requested_action and self._confabulation_correction_count == 0:
                                            print(f"[EARLY FALLBACK] User asked: '{self._last_user_request}' - Aria speaking without tool call!")
                                            self._confabulation_correction_count += 1
                                            asyncio.create_task(self._send_confabulation_correction())

                                    if self.on_speech_started:
                                        self.on_speech_started()

                        # Text output (might contain transcription in some cases)
                        if hasattr(part, 'text') and part.text:
                            self._accumulated_text += part.text
                            print(f"[Aria says]: {part.text}")

                            # CRITICAL FIX: Detect if Gemini is SAYING function calls instead of CALLING them
                            # e.g., "move_mouse(target=" or "click(target=" in the text output
                            text_lower = part.text.lower()
                            function_call_patterns = ['move_mouse(', 'click(', 'scroll(', 'open_app(', 'type_text(',
                                                      'press_key(', 'hotkey(', 'fill_field(', 'open_url(']
                            if any(pattern in text_lower for pattern in function_call_patterns):
                                logger.warning(f"[CONFABULATION] Gemini SPOKE function call instead of calling it: '{part.text}'")
                                # Clear audio queue
                                while not self._audio_output_queue.empty():
                                    try:
                                        self._audio_output_queue.get_nowait()
                                    except:
                                        break
                                # Parse and execute the function call
                                asyncio.create_task(self._parse_and_execute_spoken_function(part.text))

                            if self.on_response_text:
                                self.on_response_text(part.text)

                # Turn complete - only reset when explicitly marked complete
                if hasattr(content, 'turn_complete') and content.turn_complete:
                    queue_size = self._audio_output_queue.qsize()
                    print(f"[GeminiVoice] Turn complete, text: {len(self._accumulated_text)} chars, audio queue: {queue_size} chunks")

                    # ANTI-REPETITION: Check if Aria is repeating herself or asking useless questions
                    current_response = self._accumulated_text.strip().lower()
                    is_repetitive = False

                    if current_response and len(current_response) > 10:
                        # Check for exact/similar matches with recent responses
                        for recent in self._recent_responses:
                            if current_response == recent or current_response in recent or recent in current_response:
                                is_repetitive = True
                                break

                        # Check for unhelpful clarifying questions when user wanted an action
                        clarifying_phrases = [
                            "could you clarify", "what do you mean", "how much", "which one",
                            "can you specify", "what exactly", "please clarify", "i'm not sure",
                            "do you want me to", "should i", "are you sure", "did you mean",
                            "trouble following", "trouble understanding", "don't understand",
                            "sorry, i", "sorry i", "apologize", "having trouble", "still in development"
                        ]
                        user_request_lower = self._last_user_request.lower() if self._last_user_request else ""
                        action_keywords = ['open', 'click', 'scroll', 'type', 'press', 'close', 'new tab', 'search',
                                       'move', 'mouse', 'go to', 'navigate', 'drag', 'double']
                        user_wanted_action = any(kw in user_request_lower for kw in action_keywords)

                        if user_wanted_action and any(phrase in current_response for phrase in clarifying_phrases):
                            logger.info(f"[ANTI-REPETITION] Aria asking clarifying question instead of acting: '{current_response[:60]}...'")
                            is_repetitive = True

                        if is_repetitive:
                            logger.info(f"[ANTI-REPETITION] Stopping repetitive response")
                            while not self._audio_output_queue.empty():
                                try:
                                    self._audio_output_queue.get_nowait()
                                except:
                                    break
                            asyncio.create_task(self._send_stop_repeating())
                        else:
                            # Track this response
                            self._recent_responses.append(current_response)
                            if len(self._recent_responses) > self._max_recent_responses:
                                self._recent_responses.pop(0)

                    # CONFABULATION DETECTION: Check if Gemini claimed to do something without calling a tool
                    # This runs AFTER Gemini finishes speaking, so we have the full response
                    # NOTE: We ONLY check for confabulation (claiming success without tool call)
                    # The more aggressive "user requested action" detection is disabled as it causes garbage actions
                    accumulated_lower = self._accumulated_text.lower()

                    # Phrases that indicate Gemini claimed to perform an action
                    action_claim_phrases = [
                        'opening', 'closing', 'clicking', 'typing', 'scrolling', 'moving',
                        'i opened', 'i closed', 'i clicked', 'i typed', 'i scrolled', 'i moved',
                        "i'm opening", "i'm closing", "i'm clicking", "i'm moving",
                        'done!', 'done.', 'got it', 'there you go',
                        'it worked', 'completed', 'finished',
                        'let me try', 'i will try'
                    ]

                    claimed_action = any(phrase in accumulated_lower for phrase in action_claim_phrases)

                    if claimed_action and not self._tool_called_this_turn and self._last_user_request:
                        self._confabulation_correction_count += 1
                        logger.info(f"[CONFABULATION #{self._confabulation_correction_count}] Gemini said '{self._accumulated_text[:60]}...' but NO TOOL WAS CALLED!")

                        # Try VoiceBridge first - it's more reliable than the old fallback
                        if self._voice_bridge and self._last_action_request:
                            logger.info(f"[CONFABULATION FIX] Trying VoiceBridge for: '{self._last_action_request}'")
                            if self._try_voice_bridge(self._last_action_request):
                                logger.info(f"[CONFABULATION FIX] VoiceBridge successfully executed the action!")
                            else:
                                # VoiceBridge couldn't handle it - fall back to old method
                                if not DISABLE_CONFABULATION_FALLBACK and self._confabulation_correction_count <= self._max_confabulation_corrections:
                                    asyncio.create_task(self._send_confabulation_correction())
                        elif not DISABLE_CONFABULATION_FALLBACK and self._confabulation_correction_count <= self._max_confabulation_corrections:
                            # No VoiceBridge - use old fallback
                            asyncio.create_task(self._send_confabulation_correction())
                        else:
                            # Just log it - Gemini lied but we won't try to fix it
                            logger.info(f"[CONFABULATION] Gemini claimed action without tool call. Fallback disabled.")

                    if self._accumulated_text and self.on_response_done:
                        self.on_response_done(self._accumulated_text)

                    # Reset for next turn
                    self._accumulated_text = ""
                    self._tool_called_this_turn = False  # Reset for next turn

                    # Mark speaking as done after audio plays
                    def wait_for_audio():
                        while not self._audio_output_queue.empty():
                            time.sleep(0.1)
                        time.sleep(0.3)  # Brief pause
                        self.is_speaking = False
                        self._speech_ended_time = time.time()
                        if self.on_speech_stopped:
                            self.on_speech_stopped()
                    threading.Thread(target=wait_for_audio, daemon=True).start()

            # Handle tool calls
            if hasattr(response, 'tool_call') and response.tool_call:
                # Skip if VoiceBridge already handled this turn
                if self._tool_called_this_turn:
                    logger.info(f"[Gemini] Skipping tool call - VoiceBridge already handled this turn")
                    # Still need to send a response to Gemini so it doesn't hang
                    for func_call in response.tool_call.function_calls:
                        call_id = getattr(func_call, 'id', str(time.time()))
                        name = func_call.name
                        await self._send_tool_response(call_id, name, json.dumps({
                            "success": True,
                            "message": "Action already completed by VoiceBridge"
                        }))
                else:
                    # VoiceBridge didn't handle - proceed with Gemini tool execution
                    print(f"[Aria action]: Calling tool...")
                    self._tool_called_this_turn = True  # Mark that we actually called a tool
                    self._last_action_request = ""  # Clear action request since tool was called
                    for func_call in response.tool_call.function_calls:
                        if self.on_tool_call:
                            call_id = getattr(func_call, 'id', str(time.time()))
                            name = func_call.name
                            args = dict(func_call.args) if func_call.args else {}
                            print(f"[Aria action]: {name}({args})")

                            try:
                                result = self.on_tool_call(call_id, name, args)
                                print(f"[GeminiVoice] Tool result: {str(result)[:100]}...")
                                await self._send_tool_response(call_id, name, result)
                            except Exception as e:
                                print(f"[GeminiVoice] Tool error: {e}")
                                await self._send_tool_response(call_id, name, json.dumps({"error": str(e)}))

            # Note: Transcription is now handled at the start of this function

        except Exception as e:
            print(f"[GeminiVoice] Error handling response: {e}")
            import traceback
            traceback.print_exc()

    async def _send_tool_response(self, call_id: str, name: str, result: str) -> None:
        """Send tool result back to Gemini using the correct API."""
        if not self._active_session:
            print(f"[GeminiVoice] No active session for tool response")
            return
        try:
            # Parse result if it's JSON
            try:
                result_dict = json.loads(result)
            except:
                result_dict = {"result": result}

            print(f"[GeminiVoice] Sending tool response for {name}: {str(result_dict)[:100]}...")

            # Use the correct send_tool_response API
            function_response = types.FunctionResponse(
                id=call_id,
                name=name,
                response=result_dict
            )
            await self._active_session.send_tool_response(
                function_responses=[function_response]
            )
            print(f"[GeminiVoice] Tool response sent successfully")
        except Exception as e:
            print(f"[GeminiVoice] Error sending tool response: {e}")
            import traceback
            traceback.print_exc()

    async def _delayed_fallback(self, delay: float) -> None:
        """Wait briefly for more speech, then trigger fallback if no tool was called.

        This is the IMMEDIATE fallback - triggers as soon as we detect action words,
        rather than waiting for turn_complete. This prevents Gemini from speaking
        nonsense before we can execute the action.
        """
        await asyncio.sleep(delay)

        # Check if Gemini already called a tool (might have happened during delay)
        if self._tool_called_this_turn:
            logger.info(f"[DELAYED FALLBACK] Tool was called, skipping fallback")
            return

        # Get the full request from rolling buffer
        now = time.time()
        self._transcript_buffer = [(t, txt) for t, txt in self._transcript_buffer
                                   if now - t <= self._transcript_buffer_window]
        full_request = " ".join(txt for _, txt in self._transcript_buffer).strip()

        if not full_request:
            return

        logger.info(f"[DELAYED FALLBACK] Executing for: '{full_request}'")

        # Clear audio queue IMMEDIATELY to stop Gemini from speaking
        while not self._audio_output_queue.empty():
            try:
                self._audio_output_queue.get_nowait()
            except:
                break

        # Execute the fallback (same logic as _send_confabulation_correction)
        await self._execute_fallback_action(full_request)

    async def _execute_fallback_action(self, request: str) -> None:
        """Execute an action based on user request without Gemini tool call."""
        request_lower = request.lower()
        # Normalize to handle fragmented speech like "scro lled" -> "scrolled"
        request_normalized = request_lower.replace(' ', '')
        tool_name = None
        tool_args = {}

        # Detect app open requests
        app_patterns = {
            "chrome": ("open_app", {"app": "Google Chrome"}),
            "safari": ("open_app", {"app": "Safari"}),
            "finder": ("open_app", {"app": "Finder"}),
            "terminal": ("open_app", {"app": "Terminal"}),
            "claude": ("open_app", {"app": "Claude"}),
            "notes": ("open_app", {"app": "Notes"}),
            "messages": ("open_app", {"app": "Messages"}),
            "slack": ("open_app", {"app": "Slack"}),
            "whatsapp": ("open_app", {"app": "WhatsApp"}),
        }

        for pattern, (name, args) in app_patterns.items():
            if pattern in request_lower:
                tool_name, tool_args = name, args
                break

        # Keyboard shortcuts
        if not tool_name:
            if any(phrase in request_lower for phrase in ["new tab", "new tap", "newtab", "open tab"]):
                tool_name, tool_args = "hotkey", {"keys": ["command", "t"]}
            elif any(phrase in request_lower for phrase in ["close tab", "close tap"]):
                tool_name, tool_args = "hotkey", {"keys": ["command", "w"]}
            elif any(phrase in request_lower for phrase in ["new window"]):
                tool_name, tool_args = "hotkey", {"keys": ["command", "n"]}

        # Scroll - handle fragmented speech like "scro lled" using normalized version
        # Scroll - but NOT if user is asking to move the mouse
        mouse_words = ["mouse", "mouth", "moose", "mows"]
        is_mouse_request = any(w in request_lower or w in request_normalized for w in mouse_words)
        if not tool_name and not is_mouse_request and any(word in request_lower or word in request_normalized for word in ["scroll", "scrawl", "scrool", "scrolled", "scrolling"]):
            scroll_amount = -300  # Default down
            if any(d in request_lower or d in request_normalized for d in ["up", "top", "back"]):
                scroll_amount = 300
            import re
            amount_match = re.search(r'(\d+)\s*%?', request_lower)
            if amount_match:
                percent = int(amount_match.group(1))
                scroll_amount = int(10 * percent) if scroll_amount > 0 else int(-10 * percent)
            tool_name, tool_args = "scroll", {"amount": scroll_amount}
            logger.info(f"[IMMEDIATE FALLBACK] Detected scroll: direction={'up' if scroll_amount > 0 else 'down'}, amount={abs(scroll_amount)}")

        # Press enter
        if not tool_name and any(phrase in request_lower for phrase in ["hit enter", "press enter", "hit return"]):
            tool_name, tool_args = "press_key", {"key": "return"}

        # Move mouse - extract target from request
        # Handle speech recognition variants: "mouse" often transcribed as "mouth", "mo use", etc.
        mouse_variants = ["mouse", "mouth", "moose", "mows"]
        has_mouse = any(v in request_lower or v in request_normalized for v in mouse_variants)
        has_move = "move" in request_lower or "move" in request_normalized
        if not tool_name and has_move and has_mouse:
            import re
            import pyautogui

            # Check for directional movement: "move mouse down/up/left/right"
            # Must be specific patterns to avoid false positives like "all right"
            direction_map = {
                "down": (0, 100),
                "up": (0, -100),
                "left": (-100, 0),
                "right": (100, 0),
            }
            direction_found = None
            # Only detect direction if it's clearly a directional command
            direction_patterns = [
                r'move\s+(?:the\s+)?(?:mouse|mouth|mo\s*use|mo\s*uth)\s+(down|up|left|right)',
                r'(?:mouse|mouth|mo\s*use|mo\s*uth)\s+(down|up|left|right)',
            ]
            for pattern in direction_patterns:
                match = re.search(pattern, request_lower)
                if match:
                    direction_found = match.group(1)
                    break

            if direction_found and direction_found in direction_map:
                # Relative mouse movement - this works with coordinates
                dx, dy = direction_map[direction_found]
                current_x, current_y = pyautogui.position()
                new_x, new_y = current_x + dx, current_y + dy
                tool_name, tool_args = "move_mouse_to_coordinates", {"x": new_x, "y": new_y}
                logger.info(f"[IMMEDIATE FALLBACK] Detected relative mouse move: {direction_found} -> ({new_x}, {new_y})")
            # Note: "move mouse to X" requests should be handled by Gemini with coordinates

        # Note: Click requests should be handled by Gemini with coordinates (click_at_coordinates)

        # Execute if we found a matching action
        if tool_name and self.on_tool_call:
            try:
                # Dedupe check - use shorter cooldown for scroll (users often want to scroll repeatedly)
                action_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
                now = time.time()
                cooldown = 0.5 if tool_name == "scroll" else self._action_cooldown
                if action_key in self._recent_actions:
                    if now - self._recent_actions[action_key] < cooldown:
                        logger.info(f"[FALLBACK] Skipped duplicate: {tool_name} (cooldown: {cooldown}s)")
                        return

                call_id = f"immediate_fallback_{time.time()}"
                logger.info(f"[IMMEDIATE FALLBACK] Executing: {tool_name}({tool_args})")

                result = self.on_tool_call(call_id, tool_name, tool_args)
                logger.info(f"[IMMEDIATE FALLBACK] Result: {result}")

                self._recent_actions[action_key] = now
                self._tool_called_this_turn = True

                # Send result to Gemini
                await self._send_tool_response(call_id, tool_name, result)

                # Tell Gemini the action is done
                if self._active_session:
                    await self._active_session.send(
                        input=types.LiveClientContent(
                            turns=[
                                types.Content(
                                    role="user",
                                    parts=[types.Part(text=f"[SYSTEM: Done. {tool_name} executed successfully. Just say 'Done' or 'Got it'.]")]
                                )
                            ],
                            turn_complete=True
                        ),
                        end_of_turn=True
                    )

            except Exception as e:
                logger.error(f"[IMMEDIATE FALLBACK] Error: {e}")
                import traceback
                traceback.print_exc()

    async def _parse_and_execute_spoken_function(self, spoken_text: str) -> None:
        """Parse a function call that Gemini SAID as text and execute it.

        When Gemini outputs "move_mouse(target='icon')" as speech instead of
        actually calling the function, we parse it and execute it ourselves.
        """
        import re

        logger.info(f"[SPOKEN FUNCTION] Parsing: '{spoken_text}'")

        # Extract function name and arguments
        # Patterns: "move_mouse(target='X')", "click(target='X')", etc.
        match = re.search(r'(\w+)\s*\(\s*(?:target\s*=\s*)?["\']?([^"\']+)["\']?\s*\)', spoken_text, re.IGNORECASE)
        if not match:
            logger.warning(f"[SPOKEN FUNCTION] Could not parse: '{spoken_text}'")
            return

        func_name = match.group(1).lower()
        target = match.group(2).strip().rstrip(')')

        logger.info(f"[SPOKEN FUNCTION] Extracted: func={func_name}, target={target}")

        # Map to actual tool
        tool_name = None
        tool_args = {}

        # Note: move_mouse and click with descriptions are deprecated - Gemini should use coordinates
        if func_name in ['scroll']:
            try:
                amount = int(target) if target.lstrip('-').isdigit() else -300
            except:
                amount = -300
            tool_name = "scroll"
            tool_args = {"amount": amount}
        elif func_name in ['open_app', 'openapp']:
            tool_name = "open_app"
            tool_args = {"app": target}
        elif func_name in ['type_text', 'typetext']:
            tool_name = "type_text"
            tool_args = {"text": target}
        elif func_name in ['press_key', 'presskey']:
            tool_name = "press_key"
            tool_args = {"key": target}

        if tool_name and self.on_tool_call:
            try:
                call_id = f"spoken_func_{time.time()}"
                logger.info(f"[SPOKEN FUNCTION] Executing: {tool_name}({tool_args})")

                result = self.on_tool_call(call_id, tool_name, tool_args)
                logger.info(f"[SPOKEN FUNCTION] Result: {result}")

                self._tool_called_this_turn = True

                # Tell Gemini the action is done
                if self._active_session:
                    await self._active_session.send(
                        input=types.LiveClientContent(
                            turns=[
                                types.Content(
                                    role="user",
                                    parts=[types.Part(text=f"[SYSTEM: The {tool_name} action was executed successfully. Result: {result[:100]}. Just confirm briefly.]")]
                                )
                            ],
                            turn_complete=True
                        ),
                        end_of_turn=True
                    )

            except Exception as e:
                logger.error(f"[SPOKEN FUNCTION] Error: {e}")
                import traceback
                traceback.print_exc()

    async def _send_stop_repeating(self) -> None:
        """Tell Gemini to stop repeating itself."""
        if not self._active_session:
            return
        try:
            await self._active_session.send(
                input=types.LiveClientContent(
                    turns=[
                        types.Content(
                            role="user",
                            parts=[types.Part(text="[SYSTEM: You just said something very similar to what you said before. DO NOT REPEAT yourself. Say something NEW or just wait for the user to speak.]")]
                        )
                    ],
                    turn_complete=True
                ),
                end_of_turn=True
            )
            print(f"[GeminiVoice] Sent anti-repetition message")
        except Exception as e:
            print(f"[GeminiVoice] Error sending anti-repetition message: {e}")

    async def _send_confabulation_correction(self) -> None:
        """Execute the tool directly when Gemini fails to call it.

        This is a FALLBACK mechanism - since Gemini won't call tools,
        we parse the user's request and execute the tool ourselves.
        """
        if not self._active_session:
            return

        # Use the stored ACTION request (not the latest user request which might be "I don't see it")
        # Fall back to _last_user_request if no action request was stored
        request_to_parse = self._last_action_request if self._last_action_request else self._last_user_request
        logger.info(f"[FALLBACK] Parsing request: '{request_to_parse}' (action_request='{self._last_action_request}', user_request='{self._last_user_request}')")

        # Parse user intent and determine what tool to call
        request_lower = request_to_parse.lower()
        # Normalize to handle fragmented speech like "scro lled" -> "scrolled"
        request_normalized = request_lower.replace(' ', '')
        tool_name = None
        tool_args = {}

        # Detect "open app" requests - check for known app names
        # Map of search terms to actual app names
        app_mappings = {
            "chrome": "Google Chrome",
            "safari": "Safari",
            "finder": "Finder",
            "terminal": "Terminal",
            "claude": "Claude",
            "notes": "Notes",
            "messages": "Messages",
            "slack": "Slack",
            "whatsapp": "WhatsApp",
            "cleanmymac": "CleanMyMac",
            "clean my mac": "CleanMyMac",
            "spotify": "Spotify",
            "vscode": "Visual Studio Code",
            "vs code": "Visual Studio Code",
            "visual studio": "Visual Studio Code",
            "code": "Visual Studio Code",
            "zoom": "zoom.us",
            "discord": "Discord",
            "figma": "Figma",
            "notion": "Notion",
            "obsidian": "Obsidian",
        }

        for search_term, app_name in app_mappings.items():
            if search_term in request_lower or search_term in request_normalized:
                tool_name = "open_app"
                tool_args = {"app": app_name}
                logger.info(f"[FALLBACK] Matched app: '{search_term}' -> {app_name}")
                break

        # COMMON KEYBOARD SHORTCUTS - detect before generic "open"
        # Handle common speech recognition errors for "new tab"
        if not tool_name and any(phrase in request_lower for phrase in ["new tab", "new tap", "new town", "new tub", "newtab", "open tab", "open a tab"]):
            tool_name = "hotkey"
            tool_args = {"keys": ["command", "t"]}
            logger.info(f"[FALLBACK] Detected 'new tab' variant in: '{request_lower}' - using Cmd+T")
        if not tool_name and any(phrase in request_lower for phrase in ["close tab", "close tap", "close town", "closed tab"]):
            tool_name = "hotkey"
            tool_args = {"keys": ["command", "w"]}
            logger.info(f"[FALLBACK] Detected 'close tab' variant - using Cmd+W")
        if not tool_name and "new window" in request_lower:
            tool_name = "hotkey"
            tool_args = {"keys": ["command", "n"]}
            logger.info(f"[FALLBACK] Detected 'new window' request - using Cmd+N")

        # Handle "select all" - very common command
        if not tool_name and ("select all" in request_lower or "selectall" in request_normalized):
            tool_name = "hotkey"
            tool_args = {"keys": ["command", "a"]}
            logger.info(f"[FALLBACK] Detected 'select all' request - using Cmd+A")

        # Handle "copy" and "paste" shortcuts
        if not tool_name and "copy" in request_lower and "paste" not in request_lower:
            tool_name = "hotkey"
            tool_args = {"keys": ["command", "c"]}
            logger.info(f"[FALLBACK] Detected 'copy' request - using Cmd+C")
        if not tool_name and "paste" in request_lower:
            tool_name = "hotkey"
            tool_args = {"keys": ["command", "v"]}
            logger.info(f"[FALLBACK] Detected 'paste' request - using Cmd+V")

        # Handle "undo" and "redo"
        if not tool_name and "undo" in request_lower:
            tool_name = "hotkey"
            tool_args = {"keys": ["command", "z"]}
            logger.info(f"[FALLBACK] Detected 'undo' request - using Cmd+Z")
        if not tool_name and "redo" in request_lower:
            tool_name = "hotkey"
            tool_args = {"keys": ["command", "shift", "z"]}
            logger.info(f"[FALLBACK] Detected 'redo' request - using Cmd+Shift+Z")

        # Handle "delete" / "delete all" / "clear"
        if not tool_name and any(phrase in request_lower for phrase in ["delete all", "delete everything", "clear all", "clear everything"]):
            tool_name = "hotkey"
            tool_args = {"keys": ["command", "a"]}
            # Will need to follow with delete key - set up as two actions
            logger.info(f"[FALLBACK] Detected 'delete all' request - will select all then delete")

        if not tool_name and "open" in request_lower:
            # Try to extract app name after "open"
            import re
            match = re.search(r'open\s+(\w+)', request_lower)
            if match:
                tool_name = "open_app"
                tool_args = {"app": match.group(1).title()}

        # Detect "search" requests
        if not tool_name and "search" in request_lower and ("online" in request_lower or "web" in request_lower or "google" in request_lower):
            # Extract search query
            import re
            # Try to find what to search for
            match = re.search(r'search\s+(?:for\s+)?(?:online\s+)?(?:for\s+)?(.+)', request_lower)
            if match:
                tool_name = "web_search"
                tool_args = {"query": match.group(1).strip()}

        # Detect "scroll" requests (handle speech recognition variants and word fragmentation)
        # But NOT if user is asking to move the mouse
        mouse_words = ["mouse", "mouth", "moose", "mows"]
        is_mouse_request = any(w in request_lower or w in request_normalized for w in mouse_words)
        if not is_mouse_request and any(word in request_lower or word in request_normalized for word in ["scroll", "scrawl", "scrool", "scrolled", "scrolling"]):
            # Default is scroll down (negative = down direction in pyautogui)
            scroll_amount = -300  # Default scroll down

            # Check direction
            if any(d in request_lower for d in ["up", "top", "back"]):
                scroll_amount = 300  # Scroll up

            # Check for amount specification (scroll 30, scroll 50%, scroll a lot)
            import re
            amount_match = re.search(r'(\d+)\s*%?', request_lower)
            if amount_match:
                percent = int(amount_match.group(1))
                # Treat as percentage of screen (100% ~ 1000 pixels)
                scroll_amount = int(10 * percent) if scroll_amount > 0 else int(-10 * percent)
            elif "lot" in request_lower or "more" in request_lower:
                scroll_amount = 600 if scroll_amount > 0 else -600
            elif "little" in request_lower or "bit" in request_lower:
                scroll_amount = 150 if scroll_amount > 0 else -150

            tool_name = "scroll"
            tool_args = {"amount": scroll_amount}
            logger.info(f"[FALLBACK] Detected scroll request - direction: {'up' if scroll_amount > 0 else 'down'}, amount: {abs(scroll_amount)}, from: '{request_lower[:50]}'")

        # Detect "click" requests - click on something described
        # Use normalized check to handle fragmented speech like "clic k" -> "click"
        if not tool_name and ("click" in request_lower or "click" in request_normalized):
            import re

            # Skip if this is a complaint/negative context (user saying "you didn't click")
            # Use normalized text to handle fragments like "did n't" -> "didn't"
            normalized_for_check = request_normalized.lower() if request_normalized else request_lower
            # Also check for fragmented negatives
            negative_patterns = [
                r"didn'?t\s+click", r"don'?t\s+click", r"can'?t\s+click", r"won'?t\s+click",
                r"still\s+didn'?t", r"you\s+didn'?t", r"not\s+clicking",
                r"didn'?t\s+actually", r"failed\s+to\s+click", r"missed",
                r"did\s*n'?t", r"don\s*'?t", r"can\s*'?t",  # Handle fragmented contractions
            ]
            is_negative = any(re.search(p, normalized_for_check) for p in negative_patterns)
            # Also check raw text for fragments
            if not is_negative:
                is_negative = any(re.search(p, request_lower) for p in negative_patterns)
            if is_negative:
                logger.info(f"[FALLBACK] Skipping click - negative context detected: '{request_lower[:50]}'")
            else:
                # Extract what to click on: "click on the X", "click the button", "click in the X"
                # Try patterns in order of specificity
                patterns = [
                    r'click\s+(?:on|in)\s+(?:the\s+)?(.+)',  # "click on/in the X"
                    r'click\s+(?:the\s+)?(.+)',              # "click the X" or "click X"
                ]
                target = None
                for pattern in patterns:
                    match = re.search(pattern, request_lower)
                    if match:
                        target = match.group(1).strip()
                        break

                # If no regex match (fragmented speech), extract from normalized
                if not target and "click" in request_normalized:
                    # Remove "click" and common words, use what's left
                    target = request_lower.replace("clic", "").replace("k", "", 1).strip()
                    # Clean up
                    target = re.sub(r'^(in|on|the|a)\s+', '', target)
                    target = target.strip()

                if target:
                    # Clean up common words
                    target = re.sub(r'\s*(please|for me|now|first)\s*$', '', target)
                    target = target.strip()
                    if target:
                        tool_name = "click"
                        tool_args = {"target": target}
                        logger.info(f"[FALLBACK] Detected click request - target: '{target}'")

        # Detect "move mouse" requests - handle speech variants like "mouth"
        if not tool_name and is_mouse_request and ("move" in request_lower or "move" in request_normalized):
            import re
            import pyautogui

            # Check for directional movement: "move mouse down/up/left/right"
            # Must be specific patterns to avoid false positives like "all right"
            direction_map = {
                "down": (0, 100),
                "up": (0, -100),
                "left": (-100, 0),
                "right": (100, 0),
            }
            direction_found = None
            # Only detect direction if it's clearly a directional command
            direction_patterns = [
                r'move\s+(?:the\s+)?(?:mouse|mouth|mo\s*use|mo\s*uth)\s+(down|up|left|right)',
                r'(?:mouse|mouth|mo\s*use|mo\s*uth)\s+(down|up|left|right)',
            ]
            for pattern in direction_patterns:
                match = re.search(pattern, request_lower)
                if match:
                    direction_found = match.group(1)
                    break

            if direction_found and direction_found in direction_map:
                # Relative mouse movement
                dx, dy = direction_map[direction_found]
                current_x, current_y = pyautogui.position()
                new_x, new_y = current_x + dx, current_y + dy
                tool_name = "move_mouse_to_coordinates"
                tool_args = {"x": new_x, "y": new_y}
                logger.info(f"[FALLBACK] Detected relative mouse move: {direction_found} -> ({new_x}, {new_y})")
            else:
                # Try to extract target for "move mouse to X"
                # Patterns: "move mouse to the trash", "move the mouse to finder", etc.
                target_patterns = [
                    r'move\s+(?:the\s+)?(?:mouse|mouth|mo\s*use|mo\s*uth)\s+to\s+(?:the\s+)?(.+)',
                    r'(?:mouse|mouth)\s+to\s+(?:the\s+)?(.+)',
                ]
                for pattern in target_patterns:
                    match = re.search(pattern, request_lower)
                    if match:
                        target = match.group(1).strip()
                        # Remove trailing punctuation
                        target = target.rstrip('?.!,')
                        if target:
                            tool_name = "move_mouse_to_element"
                            tool_args = {"element": target}
                            logger.info(f"[FALLBACK] Detected move mouse to element: '{target}'")
                            break

        # Detect "type" requests - type something in the current field
        if not tool_name and ("type" in request_lower or "write" in request_lower or "enter" in request_lower):
            import re
            # Patterns: "type hello", "type 'hello world'", "type in hello", "write hello"
            # Also handle: "type X in the Y" -> fill_field

            # Check if it's "type X in the Y" (fill_field scenario)
            fill_match = re.search(r'(?:type|write|enter)\s+["\']?(.+?)["\']?\s+(?:in|into)\s+(?:the\s+)?(.+)', request_lower)
            if fill_match:
                text = fill_match.group(1).strip().strip("'\"")
                field = fill_match.group(2).strip()
                tool_name = "fill_field"
                tool_args = {"field": field, "text": text}
            else:
                # Simple type: "type hello"
                match = re.search(r'(?:type|write|enter)\s+["\']?(.+?)["\']?\s*$', request_lower)
                if match:
                    text = match.group(1).strip().strip("'\"")
                    tool_name = "type_text"
                    tool_args = {"text": text}

        # Detect "ask" or "question" in context of typing (e.g., "ask claude about X")
        if not tool_name and ("ask" in request_lower or "question" in request_lower):
            import re
            # "ask about X", "ask claude about X", "type a question about X"
            match = re.search(r'(?:ask|question)\s+(?:\w+\s+)?(?:about\s+)?(.+)', request_lower)
            if match:
                question = match.group(1).strip()
                # This is likely meant to type in an input field
                tool_name = "fill_field"
                tool_args = {"field": "text input", "text": question}

        # Detect "delete" requests - delete text in current field
        if not tool_name and ("delete" in request_lower or "remove" in request_lower or "clear" in request_lower):
            # For delete, we need to select all then delete
            # We'll handle this specially by calling two tools in sequence
            tool_name = "delete_text"  # Special handler
            tool_args = {}
            logger.info(f"[FALLBACK] Detected delete request")

        # Detect "hit enter", "press enter", "send" - press enter key
        if not tool_name and any(phrase in request_lower for phrase in ["hit enter", "press enter", "hit return", "press return"]):
            tool_name = "press_key"
            tool_args = {"key": "return"}
            logger.info(f"[FALLBACK] Detected 'enter' key request - pressing return")
        if not tool_name and "send" in request_lower:
            tool_name = "press_key"
            tool_args = {"key": "return"}
            logger.info(f"[FALLBACK] Detected 'send' request - pressing return")

        # If we identified a tool, execute it directly
        if tool_name and self.on_tool_call:
            try:
                # Check for duplicate action - use shorter cooldown for scroll (users often want to scroll repeatedly)
                action_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
                now = time.time()
                cooldown = 0.5 if tool_name == "scroll" else self._action_cooldown
                if action_key in self._recent_actions:
                    last_time = self._recent_actions[action_key]
                    if now - last_time < cooldown:
                        logger.info(f"[FALLBACK] SKIPPED duplicate action: {tool_name}({tool_args}) - executed {now - last_time:.1f}s ago, cooldown: {cooldown}s")
                        return

                call_id = f"fallback_{time.time()}"
                logger.info(f"[FALLBACK] Executing tool directly: {tool_name}({tool_args})")

                # Special handling for delete_text - need to do select all + delete
                if tool_name == "delete_text":
                    # First select all
                    result1 = self.on_tool_call(call_id + "_selectall", "hotkey", {"keys": ["command", "a"]})
                    print(f"[FALLBACK] Select all result: {result1}")
                    await asyncio.sleep(0.1)  # Small delay
                    # Then delete
                    result = self.on_tool_call(call_id + "_delete", "press_key", {"key": "delete"})
                    print(f"[FALLBACK] Delete result: {result}")
                else:
                    # Execute the tool
                    result = self.on_tool_call(call_id, tool_name, tool_args)
                logger.info(f"[FALLBACK] Tool result: {result}")

                # Mark action as completed to prevent duplicates
                self._recent_actions[action_key] = now
                self._fallback_completed = True
                self._tool_called_this_turn = True  # Mark as if tool was called

                # INTERRUPT: Clear audio queue to stop Gemini's confused response
                while not self._audio_output_queue.empty():
                    try:
                        self._audio_output_queue.get_nowait()
                    except:
                        break
                logger.info(f"[FALLBACK] Cleared audio queue - stopping Gemini's confused response")

                # Send the tool response to Gemini so it knows what happened
                await self._send_tool_response(call_id, tool_name, result)

                # Check if action actually succeeded by parsing the result
                try:
                    result_data = json.loads(result) if isinstance(result, str) else result
                    action_succeeded = result_data.get("success", False) if isinstance(result_data, dict) else False
                except:
                    action_succeeded = "success" in result.lower() if isinstance(result, str) else False

                if action_succeeded:
                    # Send a STRONG message telling Gemini the action is DONE - don't try again
                    await self._active_session.send(
                        input=types.LiveClientContent(
                            turns=[
                                types.Content(
                                    role="user",
                                    parts=[types.Part(text=f"[SYSTEM: ACTION COMPLETED SUCCESSFULLY. {tool_name} was executed and worked. Result: {result[:100]}. DO NOT try to execute this action again. Just tell the user it's done.]")]
                                )
                            ],
                            turn_complete=True
                        ),
                        end_of_turn=True
                    )
                    logger.info(f"[FALLBACK] Notified Gemini of successful execution - action complete")
                else:
                    # Action failed - tell Gemini to try with coordinates
                    await self._active_session.send(
                        input=types.LiveClientContent(
                            turns=[
                                types.Content(
                                    role="user",
                                    parts=[types.Part(text=f"[SYSTEM: ACTION FAILED. {tool_name} could not find the element. LOOK AT THE SCREEN, identify the x,y coordinates of the target, and use click_at_coordinates(x=X, y=Y) or move_mouse_to_coordinates(x=X, y=Y) instead.]")]
                                )
                            ],
                            turn_complete=True
                        ),
                        end_of_turn=True
                    )
                    logger.info(f"[FALLBACK] Action failed - told Gemini to use coordinates")
                return

            except Exception as e:
                logger.error(f"[FALLBACK] Error executing tool: {e}")
                import traceback
                traceback.print_exc()

        # If we couldn't determine the tool, send a correction message
        try:
            correction_text = (
                f"STOP. You did NOT call any tool. The user asked: '{request_to_parse}'. "
                f"Call the appropriate tool NOW. Do not respond with text - USE YOUR TOOLS."
            )

            print(f"[GeminiVoice] Sending confabulation correction: {correction_text}")

            await self._active_session.send(
                input=types.LiveClientContent(
                    turns=[
                        types.Content(
                            role="user",
                            parts=[types.Part(text=correction_text)]
                        )
                    ],
                    turn_complete=True
                ),
                end_of_turn=True
            )
            print(f"[GeminiVoice] Confabulation correction sent")

        except Exception as e:
            print(f"[GeminiVoice] Error sending confabulation correction: {e}")
            import traceback
            traceback.print_exc()

    def start_audio_input(self) -> None:
        """Start capturing audio from microphone."""
        if self._input_stream is not None and self._input_stream.is_active():
            return  # Already running

        if self._audio is None:
            self._audio = pyaudio.PyAudio()
        self.is_listening = True

        def audio_callback(in_data, frame_count, time_info, status):
            """Callback for audio input.

            Gemini Live API has built-in VAD (Voice Activity Detection)
            that handles turn-taking automatically. We just need basic
            echo prevention to avoid sending our own audio output back.
            """
            # Basic echo prevention - mute mic while playing audio
            if self.is_speaking:
                return (None, pyaudio.paContinue)

            # Send all audio to Gemini - it handles VAD internally
            if self.is_connected and self.is_listening:
                try:
                    self._audio_input_queue.put_nowait(in_data)
                except queue.Full:
                    pass
            return (None, pyaudio.paContinue)

        self._input_stream = self._audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=1024,
            stream_callback=audio_callback
        )
        self._input_stream.start_stream()
        print("[GeminiVoice] Audio input started")

    def stop_audio_input(self) -> None:
        """Stop capturing audio."""
        if self._input_stream:
            self._input_stream.stop_stream()
            self._input_stream.close()
            self._input_stream = None
        self.is_listening = False
        print("[GeminiVoice] Audio input stopped")

    def start_audio_output(self) -> None:
        """Start playing audio output."""
        if self._output_stream is not None and not self._stop_audio.is_set():
            return  # Already running

        if self._audio is None:
            self._audio = pyaudio.PyAudio()

        self._stop_audio.clear()

        self._output_stream = self._audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.config.output_sample_rate,
            output=True,
            frames_per_buffer=1024
        )

        def play_audio():
            while not self._stop_audio.is_set():
                try:
                    audio_data = self._audio_output_queue.get(timeout=0.1)
                    if self._output_stream and not self._stop_audio.is_set():
                        self._output_stream.write(audio_data)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[GeminiVoice] Audio output error: {e}")

        self._output_thread = threading.Thread(target=play_audio, daemon=True)
        self._output_thread.start()
        print("[GeminiVoice] Audio output started")

    def stop_audio_output(self) -> None:
        """Stop playing audio."""
        self._stop_audio.set()
        if self._output_thread:
            self._output_thread.join(timeout=1.0)
            self._output_thread = None
        if self._output_stream:
            self._output_stream.stop_stream()
            self._output_stream.close()
            self._output_stream = None
        print("[GeminiVoice] Audio output stopped")

    async def disconnect(self) -> None:
        """Disconnect and cleanup."""
        self.is_connected = False
        self.stop_audio_input()
        self.stop_audio_output()
        if self._audio:
            self._audio.terminate()
            self._audio = None
        self.session = None
        self._active_session = None
        print("[GeminiVoice] Disconnected")


class GeminiConversationLoop:
    """Manages a continuous voice conversation using Gemini Live API."""

    # Reconnection settings
    MAX_RECONNECT_ATTEMPTS = 10
    INITIAL_BACKOFF_SECONDS = 1.0
    MAX_BACKOFF_SECONDS = 60.0
    BACKOFF_MULTIPLIER = 2.0

    def __init__(
        self,
        api_key: str,
        config: Optional[GeminiVoiceConfig] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_handler: Optional[Callable[[str, str, Dict], str]] = None
    ):
        self.api_key = api_key
        self.config = config
        self.client = GeminiVoiceClient(api_key, config)
        self.tools = tools or []
        self.tool_handler = tool_handler
        self.is_active = False
        self._accumulated_text = ""
        self._reconnect_count = 0
        self._should_reconnect = True

        # Wire up callbacks
        self.client.on_tool_call = self._handle_tool_call
        self.client.on_transcript = self._on_transcript
        self.client.on_response_text = self._on_response_text
        self.client.on_response_done = self._on_response_done

        # External callbacks
        self.on_user_transcript: Optional[Callable[[str], None]] = None
        self.on_assistant_text: Optional[Callable[[str], None]] = None
        self.on_assistant_done: Optional[Callable[[str], None]] = None
        self.on_reconnecting: Optional[Callable[[int, float], None]] = None  # (attempt, delay)

    def _handle_tool_call(self, call_id: str, name: str, args: Dict) -> str:
        if self.tool_handler:
            return self.tool_handler(call_id, name, args)
        return json.dumps({"error": "No tool handler configured"})

    def _on_transcript(self, transcript: str) -> None:
        print(f"[User]: {transcript}")
        if self.on_user_transcript:
            self.on_user_transcript(transcript)

    def _on_response_text(self, text: str) -> None:
        self._accumulated_text += text
        if self.on_assistant_text:
            self.on_assistant_text(text)

    def _on_response_done(self, text: str) -> None:
        final_text = text or self._accumulated_text
        print(f"[Assistant]: {final_text}")
        if self.on_assistant_done:
            self.on_assistant_done(final_text)
        self._accumulated_text = ""

    async def start(self) -> bool:
        """Start the conversation loop."""
        if self.tools:
            self.client.add_tools(self.tools)

        if not await self.client.connect():
            return False

        # Start audio I/O - Gemini's built-in VAD handles turn-taking
        self.client.start_audio_output()
        self.client.start_audio_input()
        self.is_active = True
        return True

    async def run(self) -> None:
        """Run the conversation loop with auto-reconnect on failure.

        Implements exponential backoff for reconnection attempts when
        the WebSocket connection fails (e.g., 1011 service unavailable).
        """
        self._should_reconnect = True
        self._reconnect_count = 0
        backoff = self.INITIAL_BACKOFF_SECONDS

        while self._should_reconnect:
            try:
                if not self.is_active:
                    if not await self.start():
                        print("[GeminiLoop] Failed to start, will retry...")
                        await self._handle_reconnect_delay(backoff)
                        backoff = min(backoff * self.BACKOFF_MULTIPLIER, self.MAX_BACKOFF_SECONDS)
                        continue

                print("[GeminiLoop] Running continuous session...")
                self._reconnect_count = 0  # Reset on successful session start
                backoff = self.INITIAL_BACKOFF_SECONDS  # Reset backoff
                await self.client.run_session()
                print("[GeminiLoop] Session ended normally")

                # If session ended cleanly, check if we should stop
                if not self._should_reconnect:
                    break

            except asyncio.CancelledError:
                print("[GeminiLoop] Cancelled, stopping")
                self._should_reconnect = False
                break
            except Exception as e:
                error_str = str(e)
                print(f"[GeminiLoop] Error: {e}")
                import traceback
                traceback.print_exc()

                # Check for recoverable errors that warrant reconnection
                is_recoverable = any(err in error_str.lower() for err in [
                    "1011",  # Service unavailable
                    "1006",  # Abnormal closure
                    "connection",
                    "websocket",
                    "unavailable",
                    "timeout",
                    "eof",
                ])

                if is_recoverable and self._should_reconnect:
                    self._reconnect_count += 1
                    if self._reconnect_count > self.MAX_RECONNECT_ATTEMPTS:
                        print(f"[GeminiLoop] Max reconnection attempts ({self.MAX_RECONNECT_ATTEMPTS}) exceeded, stopping")
                        break

                    await self._handle_reconnect_delay(backoff)
                    backoff = min(backoff * self.BACKOFF_MULTIPLIER, self.MAX_BACKOFF_SECONDS)

                    # Clean up and prepare for reconnection
                    await self._prepare_for_reconnect()
                else:
                    # Non-recoverable error, stop
                    print(f"[GeminiLoop] Non-recoverable error, stopping")
                    break

        print("[GeminiLoop] Conversation loop exited")

    async def _handle_reconnect_delay(self, delay: float) -> None:
        """Handle reconnection delay with logging and callback."""
        print(f"[GeminiLoop] Reconnecting in {delay:.1f}s (attempt {self._reconnect_count}/{self.MAX_RECONNECT_ATTEMPTS})...")
        if self.on_reconnecting:
            self.on_reconnecting(self._reconnect_count, delay)
        await asyncio.sleep(delay)

    async def _prepare_for_reconnect(self) -> None:
        """Clean up and prepare for reconnection."""
        print("[GeminiLoop] Cleaning up for reconnection...")
        try:
            await self.client.disconnect()
        except Exception as e:
            print(f"[GeminiLoop] Error during cleanup: {e}")

        # Create a fresh client with same config
        self.client = GeminiVoiceClient(self.api_key, self.config)
        self.client.on_tool_call = self._handle_tool_call
        self.client.on_transcript = self._on_transcript
        self.client.on_response_text = self._on_response_text
        self.client.on_response_done = self._on_response_done
        self.is_active = False
        print("[GeminiLoop] Ready to reconnect")

    async def stop(self) -> None:
        """Stop the conversation loop and prevent reconnection."""
        print("[GeminiLoop] Stopping conversation loop...")
        self._should_reconnect = False
        self.is_active = False
        await self.client.disconnect()
        print("[GeminiLoop] Stopped")


# Aria tool definitions for Gemini Live API
ARIA_GEMINI_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "look_at_screen",
        "description": "Look at the user's screen. ALWAYS call this first when asked about the screen.",
        "parameters": {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "What to focus on"
                }
            }
        }
    },
    {
        "name": "execute_task",
        "description": "Execute a task using vision to plan and verify.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "What to accomplish"}
            },
            "required": ["task"]
        }
    },
    {
        "name": "type_text",
        "description": "Type text at cursor position.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to type"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "fill_field",
        "description": "BEST FOR FORM FIELDS! Click on an input field AND type text into it in one action. Use this instead of separate click+type. Examples: fill_field(field='First Name', text='John'), fill_field(field='Email', text='test@example.com'), fill_field(field='search box', text='query')",
        "parameters": {
            "type": "object",
            "properties": {
                "field": {"type": "string", "description": "Description of the input field to click (e.g., 'First Name', 'Email', 'search box')"},
                "text": {"type": "string", "description": "Text to type into the field"}
            },
            "required": ["field", "text"]
        }
    },
    {
        "name": "press_key",
        "description": "Press a key (enter, tab, escape, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key to press"}
            },
            "required": ["key"]
        }
    },
    {
        "name": "hotkey",
        "description": "Press a keyboard shortcut. USE THIS FOR: new tab (['command','t']), close tab (['command','w']), new window (['command','n']), copy (['command','c']), paste (['command','v']), undo (['command','z']), save (['command','s']), find (['command','f']), select all (['command','a']), address bar (['command','l']). This is the fastest way to control apps.",
        "parameters": {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keys to press together, e.g. ['command', 't'] for new tab"
                }
            },
            "required": ["keys"]
        }
    },
    {
        "name": "open_app",
        "description": "Open an application by name",
        "parameters": {
            "type": "object",
            "properties": {
                "app": {"type": "string", "description": "App name"}
            },
            "required": ["app"]
        }
    },
    {
        "name": "open_url",
        "description": "Open a URL in browser",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to open"}
            },
            "required": ["url"]
        }
    },
    {
        "name": "scroll",
        "description": "Scroll. Positive = up, negative = down.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "integer", "description": "Scroll amount"}
            },
            "required": ["amount"]
        }
    },
    {
        "name": "remember",
        "description": "Store information in memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "Fact to remember"},
                "category": {"type": "string", "description": "Category"}
            },
            "required": ["fact"]
        }
    },
    {
        "name": "recall",
        "description": "Search memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "web_search",
        "description": "Search the web for current information. Use for questions about current events, weather, prices, facts, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "double_click",
        "description": "Double-click on a UI element by description. Use to open files, apps, or activate items.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "What to double-click"}
            },
            "required": ["target"]
        }
    },
    {
        "name": "right_click",
        "description": "Right-click on a UI element to open context menu.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "What to right-click"}
            },
            "required": ["target"]
        }
    },
    {
        "name": "drag",
        "description": "Drag from one element to another. Use for drag-and-drop, resizing, or moving items.",
        "parameters": {
            "type": "object",
            "properties": {
                "from_target": {"type": "string", "description": "Element to drag from"},
                "to_target": {"type": "string", "description": "Element to drag to"}
            },
            "required": ["from_target", "to_target"]
        }
    },
    {
        "name": "get_mouse_position",
        "description": "Get the current mouse cursor position on screen.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "click_at_coordinates",
        "description": "Click at x,y coordinates. Look at the screenshot, find the element's center, provide its pixel coordinates.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coordinate (pixels from left edge)"},
                "y": {"type": "integer", "description": "Y coordinate (pixels from top edge)"},
                "button": {"type": "string", "description": "Button: left, right, or middle", "default": "left"}
            },
            "required": ["x", "y"]
        }
    },
    {
        "name": "move_mouse_to_coordinates",
        "description": "Move mouse to x,y coordinates. Only use if you know exact pixel coordinates.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coordinate (pixels from left edge)"},
                "y": {"type": "integer", "description": "Y coordinate (pixels from top edge)"}
            },
            "required": ["x", "y"]
        }
    },
    {
        "name": "move_mouse_to_element",
        "description": "Move mouse to a UI element by name. PREFERRED for dock icons and apps. Uses macOS Accessibility API for accurate positioning. Examples: 'Chrome', 'Finder', 'Docker', 'Trash'",
        "parameters": {
            "type": "object",
            "properties": {
                "element": {"type": "string", "description": "Name of the element (e.g., 'Chrome', 'Finder icon', 'Trash in dock')"}
            },
            "required": ["element"]
        }
    },
    {
        "name": "click_element",
        "description": "Click on a DOCK ICON by name. ONLY for dock icons at bottom of screen. Uses macOS Accessibility API.",
        "parameters": {
            "type": "object",
            "properties": {
                "element": {"type": "string", "description": "Name of the dock icon to click"}
            },
            "required": ["element"]
        }
    },
    {
        "name": "click_target",
        "description": "PREFERRED for clicking UI elements. Describe what to click and Claude's vision will find it precisely. Use for buttons, links, text, icons, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Description of what to click (e.g., 'the send button', 'the blue Submit link', 'the X close icon')"},
                "button": {"type": "string", "description": "Mouse button: left, right, or middle", "default": "left"}
            },
            "required": ["target"]
        }
    },
    {
        "name": "move_to_target",
        "description": "PREFERRED for moving mouse to UI elements. Describe where to move and Claude's vision will find it precisely.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Description of where to move mouse (e.g., 'the user name in top right', 'the search box', 'the red X button')"}
            },
            "required": ["target"]
        }
    },
    {
        "name": "list_browser_tabs",
        "description": "List all open browser tabs. Returns tab titles. Use when asked 'what tabs are open', 'show tabs', 'which tabs', etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "browser": {"type": "string", "description": "Browser name: 'Google Chrome' or 'Safari'. Default: Google Chrome"}
            }
        }
    },
    {
        "name": "close_browser_tab",
        "description": "Close a browser tab. Can close by tab title/URL match, or close current tab. MUCH MORE RELIABLE than keyboard shortcuts for closing specific tabs.",
        "parameters": {
            "type": "object",
            "properties": {
                "tab": {"type": "string", "description": "Part of tab title or URL to match. Leave empty to close current tab."},
                "browser": {"type": "string", "description": "Browser name: 'Google Chrome' or 'Safari'. Default: Google Chrome"}
            }
        }
    },
    {
        "name": "switch_browser_tab",
        "description": "Switch to a browser tab by title or URL match. MUCH MORE RELIABLE than clicking on tabs.",
        "parameters": {
            "type": "object",
            "properties": {
                "tab": {"type": "string", "description": "Part of tab title or URL to match"},
                "browser": {"type": "string", "description": "Browser name: 'Google Chrome' or 'Safari'. Default: Google Chrome"}
            },
            "required": ["tab"]
        }
    },
    {
        "name": "close_tab_by_position",
        "description": "Close a browser tab by its position (e.g., 'second from right', 'third tab', 'last tab'). Use when user refers to tabs by position rather than name.",
        "parameters": {
            "type": "object",
            "properties": {
                "position": {"type": "string", "description": "Position description like 'second from right', 'third tab', 'last tab', 'first tab'"},
                "browser": {"type": "string", "description": "Browser name: 'Google Chrome' or 'Safari'. Default: Google Chrome"}
            },
            "required": ["position"]
        }
    },
    {
        "name": "schedule_research",
        "description": "Schedule proactive learning on a topic. Aria will automatically research this periodically to build expertise. Use when asked to 'learn about', 'research', 'keep up with', or 'stay updated on' something.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Short topic name (e.g., 'TikTok trends', 'Mac shortcuts')"},
                "query": {"type": "string", "description": "Full search query to research"},
                "interval_hours": {"type": "integer", "description": "How often to research (default: 168 = weekly)"}
            },
            "required": ["topic", "query"]
        }
    },
    {
        "name": "list_research",
        "description": "List all scheduled proactive research tasks. Shows what Aria is continuously learning about.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]


def create_gemini_tool_handler(control_module, memory_module, vision_module=None) -> Callable[[str, str, Dict], str]:
    """Create a tool handler for Gemini."""
    from .action_executor import get_action_executor

    def handle_tool(call_id: str, name: str, args: Dict) -> str:
        try:
            if name == "look_at_screen":
                if vision_module:
                    result = vision_module.capture_to_base64_with_size()
                    if result:
                        image_b64, (width, height) = result
                        from .lazy_anthropic import get_client
                        from .config import ANTHROPIC_API_KEY, CLAUDE_MODEL
                        client = get_client(ANTHROPIC_API_KEY)

                        focus = args.get("focus", "the entire screen")
                        response = client.messages.create(
                            model=CLAUDE_MODEL,
                            max_tokens=1024,
                            messages=[{
                                "role": "user",
                                "content": [
                                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                                    {"type": "text", "text": f"Describe what you see, focusing on: {focus}. Be concise."}
                                ]
                            }]
                        )
                        return json.dumps({"description": response.content[0].text, "screen_size": f"{width}x{height}"})
                return json.dumps({"error": "Vision not available"})

            elif name == "execute_task":
                executor = get_action_executor(control_module)
                result = executor.execute_task(args["task"])
                return json.dumps({"success": result.success, "message": result.message})

            elif name == "click":
                if "target" in args:
                    executor = get_action_executor(control_module)
                    result = executor.click_element(args["target"])
                    return json.dumps({"success": result.success, "message": result.message})
                return json.dumps({"error": "No target specified"})

            elif name == "type_text":
                control_module.type_text(args["text"])
                return json.dumps({"success": True})

            elif name == "press_key":
                control_module.press_key(args["key"])
                return json.dumps({"success": True})

            elif name == "hotkey":
                control_module.hotkey(args["keys"])
                return json.dumps({"success": True})

            elif name == "open_app":
                control_module.open_app(args["app"])
                return json.dumps({"success": True})

            elif name == "open_url":
                control_module.open_url(args["url"])
                return json.dumps({"success": True})

            elif name == "scroll":
                control_module.scroll(args["amount"])
                return json.dumps({"success": True})

            elif name == "remember":
                memory_module.add(args["fact"], category=args.get("category", "other"))
                return json.dumps({"success": True})

            elif name == "recall":
                results = memory_module.search(args["query"], n_results=5)
                return json.dumps({"results": results})

            else:
                return json.dumps({"error": f"Unknown tool: {name}"})

        except Exception as e:
            return json.dumps({"error": str(e)})

    return handle_tool
