"""
OpenAI Realtime Voice for Aria.

Provides sub-second voice-to-voice latency using OpenAI's Realtime API.
This replaces the traditional Whisper STT -> Claude -> TTS pipeline with
a single WebSocket connection for real-time audio streaming.
"""

import asyncio
import base64
import json
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pyaudio

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketClientProtocol = None


@dataclass
class RealtimeConfig:
    """Configuration for OpenAI Realtime API.

    Attributes:
        model: The model to use for realtime voice interactions.
        voice: The voice to use for audio output (alloy, echo, fable, onyx, nova, shimmer).
        input_audio_format: Audio format for input (pcm16 or g711_ulaw or g711_alaw).
        output_audio_format: Audio format for output (pcm16 or g711_ulaw or g711_alaw).
        sample_rate: Audio sample rate in Hz.
        vad_threshold: Voice activity detection threshold (0.0 to 1.0).
        silence_duration_ms: Duration of silence to detect end of speech.
        prefix_padding_ms: Padding before speech detection.
        instructions: System instructions for the assistant.
    """
    model: str = "gpt-4o-realtime-preview-2024-12-17"
    voice: str = "alloy"
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm16"
    sample_rate: int = 24000
    vad_threshold: float = 0.5
    silence_duration_ms: int = 500
    prefix_padding_ms: int = 300
    instructions: str = ""


@dataclass
class RealtimeToolDefinition:
    """Definition for a function calling tool.

    Attributes:
        name: The name of the function.
        description: Description of what the function does.
        parameters: JSON Schema for the function parameters.
    """
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API format."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


class RealtimeVoiceClient:
    """OpenAI Realtime API client for voice interaction.

    This client manages a WebSocket connection to OpenAI's Realtime API,
    handling bidirectional audio streaming with voice activity detection
    and function calling support.

    Usage:
        client = RealtimeVoiceClient(api_key="sk-...")
        await client.connect()
        client.start_audio_input()
        client.start_audio_output()
        await client.listen_events()
    """

    REALTIME_URL = "wss://api.openai.com/v1/realtime"

    def __init__(self, api_key: str, config: Optional[RealtimeConfig] = None):
        """Initialize the Realtime Voice Client.

        Args:
            api_key: OpenAI API key.
            config: Configuration for the realtime session.
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError(
                "websockets package is required for Realtime Voice. "
                "Install with: pip install websockets"
            )

        self.api_key = api_key
        self.config = config or RealtimeConfig()
        self.ws: Optional[WebSocketClientProtocol] = None

        # Audio handling
        self._audio: Optional[pyaudio.PyAudio] = None
        self._input_stream: Optional[pyaudio.Stream] = None
        self._output_stream: Optional[pyaudio.Stream] = None
        self._audio_output_queue: queue.Queue = queue.Queue()
        self._audio_input_loop: Optional[asyncio.AbstractEventLoop] = None

        # State
        self.is_connected = False
        self.is_speaking = False
        self.is_listening = False
        self._input_muted = False  # Soft mute - still listens but filters echo
        self._recent_assistant_text = ""  # Track what Aria said for echo filtering
        self._speech_ended_time: float = 0  # When Aria stopped speaking (for cooldown)
        self._echo_cooldown_seconds: float = 2.0  # Filter echo for this long after speaking
        self.current_response_id: Optional[str] = None
        self._stop_audio = threading.Event()
        self._output_thread: Optional[threading.Thread] = None

        # Interrupt detection via audio energy
        self._baseline_energy: float = 0.0  # Baseline audio energy (Aria's output level)
        self._interrupt_threshold: float = 5.0  # Energy must be 5x baseline to be interrupt
        self._min_baseline: float = 500.0  # Minimum baseline before interrupt detection active
        self._energy_samples: list = []  # Recent energy samples for baseline
        self._pending_interrupt: bool = False  # Flag when interrupt detected
        self._samples_before_interrupt: int = 30  # Need this many samples before detecting interrupts

        # Callbacks
        self.on_transcript: Optional[Callable[[str], None]] = None
        self.on_response_text: Optional[Callable[[str], None]] = None
        self.on_response_done: Optional[Callable[[str], None]] = None
        self.on_tool_call: Optional[Callable[[str, str, Dict], str]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_speech_started: Optional[Callable[[], None]] = None
        self.on_speech_stopped: Optional[Callable[[], None]] = None
        self.on_input_speech_started: Optional[Callable[[], None]] = None
        self.on_input_speech_stopped: Optional[Callable[[], None]] = None

    async def connect(self) -> bool:
        """Establish WebSocket connection to Realtime API.

        Returns:
            True if connection successful, False otherwise.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }

        url = f"{self.REALTIME_URL}?model={self.config.model}"

        try:
            # Try both header parameter names for websockets compatibility
            # websockets < 11: extra_headers, websockets >= 11: additional_headers
            try:
                self.ws = await websockets.connect(
                    url,
                    additional_headers=headers,  # websockets >= 11
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10
                )
            except TypeError:
                # Fallback for older websockets versions
                self.ws = await websockets.connect(
                    url,
                    extra_headers=headers,  # websockets < 11
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10
                )
            self.is_connected = True
            await self._configure_session()
            print("[RealtimeVoice] Connected to OpenAI Realtime API")
            return True
        except Exception as e:
            print(f"[RealtimeVoice] Failed to connect: {e}")
            if self.on_error:
                self.on_error(f"Connection failed: {e}")
            return False

    async def _configure_session(self) -> None:
        """Configure the realtime session with initial settings."""
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": self.config.voice,
                "input_audio_format": self.config.input_audio_format,
                "output_audio_format": self.config.output_audio_format,
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": self.config.vad_threshold,
                    "silence_duration_ms": self.config.silence_duration_ms,
                    "prefix_padding_ms": self.config.prefix_padding_ms,
                },
                "input_audio_transcription": {
                    "model": "whisper-1"
                }
            }
        }

        # Add instructions if provided
        if self.config.instructions:
            session_config["session"]["instructions"] = self.config.instructions

        await self._send(session_config)

    async def add_tools(self, tools: List[RealtimeToolDefinition]) -> None:
        """Add function calling tools to the session.

        Args:
            tools: List of tool definitions to add.
        """
        tool_dicts = [t.to_dict() if isinstance(t, RealtimeToolDefinition) else t for t in tools]
        await self._send({
            "type": "session.update",
            "session": {
                "tools": tool_dicts,
                "tool_choice": "auto"
            }
        })

    async def send_audio(self, audio_data: bytes) -> None:
        """Send audio chunk to the API.

        Args:
            audio_data: Raw PCM16 audio bytes.
        """
        if not self.is_connected or not self.ws:
            return

        await self._send({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio_data).decode('utf-8')
        })

    async def commit_audio(self) -> None:
        """Commit the audio buffer to trigger processing."""
        await self._send({
            "type": "input_audio_buffer.commit"
        })

    async def clear_audio_buffer(self) -> None:
        """Clear the input audio buffer."""
        await self._send({
            "type": "input_audio_buffer.clear"
        })

    async def cancel_response(self) -> None:
        """Cancel the current response (for interruption handling)."""
        if self.current_response_id and self.is_speaking:
            await self._send({
                "type": "response.cancel"
            })
            self.is_speaking = False
            self.current_response_id = None
            # Clear the audio output queue
            while not self._audio_output_queue.empty():
                try:
                    self._audio_output_queue.get_nowait()
                except queue.Empty:
                    break

    async def send_text(self, text: str) -> None:
        """Send a text message to the conversation.

        Args:
            text: The text message to send.
        """
        await self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text
                    }
                ]
            }
        })
        await self._send({"type": "response.create"})

    async def send_tool_result(self, call_id: str, result: str) -> None:
        """Send tool call result back to API.

        Args:
            call_id: The ID of the tool call.
            result: The result of the tool execution (JSON string).
        """
        await self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result
            }
        })
        await self._send({"type": "response.create"})

    async def _send(self, message: Dict[str, Any]) -> None:
        """Send a message to the WebSocket.

        Args:
            message: The message to send.
        """
        if self.ws:
            await self.ws.send(json.dumps(message))

    async def listen_events(self) -> None:
        """Listen for events from the Realtime API.

        This method runs continuously, processing incoming events.
        It should be run in an asyncio task.
        """
        if not self.ws:
            return

        try:
            async for message in self.ws:
                try:
                    event = json.loads(message)
                    await self._handle_event(event)
                except json.JSONDecodeError:
                    continue
        except websockets.exceptions.ConnectionClosed as e:
            print(f"[RealtimeVoice] Connection closed: {e}")
            self.is_connected = False
        except Exception as e:
            print(f"[RealtimeVoice] Error in event loop: {e}")
            if self.on_error:
                self.on_error(str(e))

    async def _handle_event(self, event: Dict[str, Any]) -> None:
        """Handle incoming events from the API.

        Args:
            event: The event data.
        """
        event_type = event.get("type", "")

        # Audio output events
        if event_type == "response.audio.delta":
            audio_data = base64.b64decode(event.get("delta", ""))
            self._audio_output_queue.put(audio_data)
            if not self.is_speaking:
                self.is_speaking = True
                # Reset energy tracking for interrupt detection
                self._energy_samples.clear()
                self._baseline_energy = 0
                self._pending_interrupt = False
                if self.on_speech_started:
                    self.on_speech_started()

        elif event_type == "response.audio.done":
            # Don't set is_speaking=False yet - audio is still in queue being played
            # Start a thread to wait for queue to drain, then set is_speaking=False
            def wait_for_audio_queue():
                # Wait for audio queue to be empty (audio finished playing)
                while not self._audio_output_queue.empty():
                    time.sleep(0.1)
                # Add extra delay for speaker latency
                time.sleep(0.3)
                # Now safe to say we're not speaking
                self.is_speaking = False
                self._speech_ended_time = time.time()
                self._energy_samples.clear()
                self._baseline_energy = 0
                print("[RealtimeVoice] Audio playback finished")
                if self.on_speech_stopped:
                    self.on_speech_stopped()
            threading.Thread(target=wait_for_audio_queue, daemon=True).start()

        # Audio transcript events - THIS is what Aria says in audio form
        # This is critical for echo filtering!
        elif event_type == "response.audio_transcript.delta":
            delta = event.get("delta", "")
            self._recent_assistant_text += delta  # Track for echo filtering
            if self.on_response_text:
                self.on_response_text(delta)

        elif event_type == "response.audio_transcript.done":
            transcript = event.get("transcript", "")
            print(f"[RealtimeVoice] Assistant said: {transcript[:80]}...")

        # Text response events (for text-only mode, less common)
        elif event_type == "response.text.delta":
            delta = event.get("delta", "")
            self._recent_assistant_text += delta  # Track for echo filtering
            if self.on_response_text:
                self.on_response_text(delta)

        elif event_type == "response.text.done":
            pass  # Handled by response.done

        # Transcription events
        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            if self.on_transcript and transcript:
                is_interrupt = self._is_interrupt_command(transcript)

                # If Aria is currently speaking and this ISN'T an interrupt command, it's echo
                if self.is_speaking and not is_interrupt:
                    print(f"[RealtimeVoice] Filtered (speaking): {transcript[:50]}...")
                    return

                # If we're in cooldown period after speaking, filter non-interrupts
                if self._in_echo_cooldown() and not is_interrupt:
                    # Also check if it matches what Aria said
                    if self._is_echo(transcript):
                        print(f"[RealtimeVoice] Filtered (cooldown+echo): {transcript[:50]}...")
                        return
                    # Even if not exact match, short phrases during cooldown are likely echo
                    if len(transcript.strip()) < 20:
                        print(f"[RealtimeVoice] Filtered (cooldown+short): {transcript[:50]}...")
                        return

                # Standard echo check
                if self._is_echo(transcript):
                    print(f"[RealtimeVoice] Filtered echo: {transcript[:50]}...")
                    return

                self.on_transcript(transcript)

        # Input speech detection
        elif event_type == "input_audio_buffer.speech_started":
            if self.on_input_speech_started:
                self.on_input_speech_started()
            # Cancel current response if user starts speaking (interruption)
            if self.is_speaking and self.current_response_id:
                await self.cancel_response()

        elif event_type == "input_audio_buffer.speech_stopped":
            if self.on_input_speech_stopped:
                self.on_input_speech_stopped()

        # Function calling events
        elif event_type == "response.function_call_arguments.done":
            if self.on_tool_call:
                call_id = event.get("call_id", "")
                name = event.get("name", "")
                try:
                    args = json.loads(event.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                # Execute tool and send result
                try:
                    result = self.on_tool_call(call_id, name, args)
                    await self.send_tool_result(call_id, result)
                except Exception as e:
                    await self.send_tool_result(call_id, json.dumps({"error": str(e)}))

        # Response lifecycle events
        elif event_type == "response.created":
            self.current_response_id = event.get("response", {}).get("id")
            # Set speaking flag EARLY to prevent echo from being sent
            # This is before audio actually starts, but prevents race condition
            self.is_speaking = True
            self._energy_samples.clear()
            self._baseline_energy = 0
            # Clear any audio that was buffered before we started responding
            await self.clear_audio_buffer()
            print("[RealtimeVoice] Response starting - muting input")

        elif event_type == "response.done":
            response = event.get("response", {})
            self.current_response_id = None
            # Note: is_speaking is managed by audio.done handler which waits for queue to drain

            # Extract final text/transcript if available
            if self.on_response_done:
                output = response.get("output", [])
                final_text = ""
                for item in output:
                    if item.get("type") == "message":
                        for content in item.get("content", []):
                            # Check both text and audio transcript
                            if content.get("type") == "text":
                                final_text += content.get("text", "")
                            elif content.get("type") == "audio":
                                final_text += content.get("transcript", "")
                if final_text:
                    self.on_response_done(final_text)

            # Clear echo buffer after delay (keep it for a bit to catch delayed transcriptions)
            def clear_echo_buffer():
                time.sleep(4)  # Longer delay - transcriptions can be delayed
                self._recent_assistant_text = ""
            threading.Thread(target=clear_echo_buffer, daemon=True).start()

        # Error events
        elif event_type == "error":
            error_info = event.get("error", {})
            error_msg = error_info.get("message", "Unknown error")
            error_code = error_info.get("code", "unknown")
            # Ignore harmless cancel errors (race condition when response finishes during cancel)
            if error_code == "response_cancel_not_active":
                return  # Silently ignore - this is expected behavior
            print(f"[RealtimeVoice] API error ({error_code}): {error_msg}")
            if self.on_error:
                self.on_error(f"{error_code}: {error_msg}")

        # Session events
        elif event_type == "session.created":
            print("[RealtimeVoice] Session created")

        elif event_type == "session.updated":
            print("[RealtimeVoice] Session updated")

    def start_audio_input(self) -> None:
        """Start capturing audio from microphone."""
        if self._input_stream is not None:
            return

        self._audio = pyaudio.PyAudio()
        self._audio_input_loop = asyncio.get_event_loop()
        self.is_listening = True

        def audio_callback(in_data, frame_count, time_info, status):
            """Callback for audio input."""
            # Don't send audio while Aria is speaking - prevents echo feedback
            # NOTE: Interrupts are disabled until we implement proper echo cancellation
            # For now, wait for Aria to finish speaking before talking
            if self.is_speaking:
                return (None, pyaudio.paContinue)

            # Cooldown period after speaking - wait for echo to dissipate
            if self._speech_ended_time > 0:
                elapsed = time.time() - self._speech_ended_time
                if elapsed < 1.0:  # 1 second cooldown
                    return (None, pyaudio.paContinue)

            # Normal operation - send audio to server
            if self._audio_input_loop and self.is_connected and self.is_listening:
                try:
                    if not self._audio_input_loop.is_closed():
                        asyncio.run_coroutine_threadsafe(
                            self.send_audio(in_data),
                            self._audio_input_loop
                        )
                except RuntimeError:
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
        print("[RealtimeVoice] Audio input started")

    def stop_audio_input(self) -> None:
        """Stop capturing audio from microphone."""
        if self._input_stream:
            self._input_stream.stop_stream()
            self._input_stream.close()
            self._input_stream = None
        self.is_listening = False
        print("[RealtimeVoice] Audio input stopped")

    def _mute_input(self) -> None:
        """Temporarily mute microphone input (prevents feedback when speaking)."""
        self._input_muted = True

    def _unmute_input(self) -> None:
        """Unmute microphone input after speaking."""
        self._input_muted = False

    async def _handle_interrupt(self) -> None:
        """Handle user interrupt during Aria's speech."""
        if not self._pending_interrupt:
            return

        print("[RealtimeVoice] Processing interrupt...")
        self._pending_interrupt = False

        # Cancel current response
        await self.cancel_response()

        # Clear audio output queue (stop playing)
        while not self._audio_output_queue.empty():
            try:
                self._audio_output_queue.get_nowait()
            except queue.Empty:
                break

        # Reset state
        self.is_speaking = False
        self._speech_ended_time = time.time()
        self._energy_samples.clear()
        self._baseline_energy = 0

        # Clear input buffer
        await self.clear_audio_buffer()

        print("[RealtimeVoice] Interrupt handled - listening for user")

    def _in_echo_cooldown(self) -> bool:
        """Check if we're in the echo cooldown period after Aria stopped speaking.

        Returns:
            True if we're still in the cooldown period.
        """
        if self._speech_ended_time == 0:
            return False
        elapsed = time.time() - self._speech_ended_time
        return elapsed < self._echo_cooldown_seconds

    def _is_interrupt_command(self, transcript: str) -> bool:
        """Check if a transcript is an interrupt command.

        These commands should always be passed through even if Aria is speaking.

        Args:
            transcript: The transcribed text.

        Returns:
            True if this is an interrupt command.
        """
        if not transcript:
            return False

        transcript_lower = transcript.lower().strip()

        interrupt_commands = [
            "stop", "aria", "hey aria", "wait", "pause", "hold on",
            "never mind", "cancel", "shut up", "quiet", "enough",
            "okay stop", "ok stop", "that's enough", "thanks", "thank you",
            "no", "actually", "wait wait", "hold", "shh", "shush"
        ]

        for cmd in interrupt_commands:
            if cmd in transcript_lower:
                return True

        return False

    def _is_echo(self, transcript: str) -> bool:
        """Check if a transcript is likely echo of Aria's own speech.

        This allows filtering out microphone pickup of Aria's speaker output
        while still allowing user interruptions like "stop" or "aria".

        Args:
            transcript: The transcribed user speech.

        Returns:
            True if this appears to be echo, False if it's real user input.
        """
        if not transcript or not self._recent_assistant_text:
            return False

        transcript_lower = transcript.lower().strip()
        assistant_lower = self._recent_assistant_text.lower()

        # Always allow interrupt commands through (these are NOT echo)
        interrupt_commands = [
            "stop", "aria", "hey aria", "wait", "pause", "hold on",
            "never mind", "cancel", "shut up", "quiet", "enough",
            "okay stop", "ok stop", "that's enough", "thanks"
        ]
        for cmd in interrupt_commands:
            if cmd in transcript_lower:
                return False

        # Check if transcript appears in what Aria recently said
        # Use substring matching since transcription may capture partial phrases
        if len(transcript_lower) < 5:
            # Very short phrases are ambiguous, let them through
            return False

        # Check for significant overlap
        # If more than 60% of the transcript words appear in assistant text, it's echo
        transcript_words = set(transcript_lower.split())
        assistant_words = set(assistant_lower.split())

        if not transcript_words:
            return False

        overlap = transcript_words & assistant_words
        overlap_ratio = len(overlap) / len(transcript_words)

        if overlap_ratio > 0.6:
            return True

        # Also check for direct substring match (captures exact phrases)
        # Remove punctuation for comparison
        clean_transcript = re.sub(r'[^\w\s]', '', transcript_lower)
        clean_assistant = re.sub(r'[^\w\s]', '', assistant_lower)

        if len(clean_transcript) > 10 and clean_transcript in clean_assistant:
            return True

        return False

    def start_audio_output(self) -> None:
        """Start playing audio output."""
        if self._output_stream is not None:
            return

        if self._audio is None:
            self._audio = pyaudio.PyAudio()

        self._stop_audio.clear()

        self._output_stream = self._audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.config.sample_rate,
            output=True,
            frames_per_buffer=1024
        )

        def play_audio():
            """Thread function to play audio from queue."""
            while not self._stop_audio.is_set():
                try:
                    audio_data = self._audio_output_queue.get(timeout=0.1)
                    if self._output_stream and not self._stop_audio.is_set():
                        self._output_stream.write(audio_data)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[RealtimeVoice] Audio output error: {e}")

        self._output_thread = threading.Thread(target=play_audio, daemon=True)
        self._output_thread.start()
        print("[RealtimeVoice] Audio output started")

    def stop_audio_output(self) -> None:
        """Stop playing audio output."""
        self._stop_audio.set()
        if self._output_thread:
            self._output_thread.join(timeout=1.0)
            self._output_thread = None
        if self._output_stream:
            self._output_stream.stop_stream()
            self._output_stream.close()
            self._output_stream = None
        # Clear remaining audio
        while not self._audio_output_queue.empty():
            try:
                self._audio_output_queue.get_nowait()
            except queue.Empty:
                break
        print("[RealtimeVoice] Audio output stopped")

    async def disconnect(self) -> None:
        """Disconnect from Realtime API and cleanup resources."""
        self.stop_audio_input()
        self.stop_audio_output()

        if self._audio:
            self._audio.terminate()
            self._audio = None

        if self.ws:
            await self.ws.close()
            self.ws = None

        self.is_connected = False
        print("[RealtimeVoice] Disconnected")


class RealtimeConversationLoop:
    """Manages a continuous voice conversation using the Realtime API.

    This provides a higher-level interface for running voice conversations,
    similar to ConversationLoop but using the Realtime API.
    """

    def __init__(
        self,
        api_key: str,
        config: Optional[RealtimeConfig] = None,
        tools: Optional[List[RealtimeToolDefinition]] = None,
        tool_handler: Optional[Callable[[str, str, Dict], str]] = None
    ):
        """Initialize the conversation loop.

        Args:
            api_key: OpenAI API key.
            config: Realtime configuration.
            tools: List of tools to make available.
            tool_handler: Function to handle tool calls.
        """
        self.client = RealtimeVoiceClient(api_key, config)
        self.tools = tools or []
        self.tool_handler = tool_handler

        # Track conversation state
        self.is_active = False
        self._accumulated_text = ""

        # Wire up callbacks
        self.client.on_tool_call = self._handle_tool_call
        self.client.on_transcript = self._on_transcript
        self.client.on_response_text = self._on_response_text
        self.client.on_response_done = self._on_response_done

        # External callbacks
        self.on_user_transcript: Optional[Callable[[str], None]] = None
        self.on_assistant_text: Optional[Callable[[str], None]] = None
        self.on_assistant_done: Optional[Callable[[str], None]] = None

    def _handle_tool_call(self, call_id: str, name: str, args: Dict) -> str:
        """Handle tool call from the API."""
        if self.tool_handler:
            return self.tool_handler(call_id, name, args)
        return json.dumps({"error": "No tool handler configured"})

    def _on_transcript(self, transcript: str) -> None:
        """Handle user transcript."""
        print(f"[User]: {transcript}")
        if self.on_user_transcript:
            self.on_user_transcript(transcript)

    def _on_response_text(self, text: str) -> None:
        """Handle response text delta."""
        self._accumulated_text += text
        if self.on_assistant_text:
            self.on_assistant_text(text)

    def _on_response_done(self, text: str) -> None:
        """Handle response completion."""
        final_text = text or self._accumulated_text
        print(f"[Assistant]: {final_text}")
        if self.on_assistant_done:
            self.on_assistant_done(final_text)
        self._accumulated_text = ""

    async def start(self) -> bool:
        """Start the conversation loop.

        Returns:
            True if started successfully.
        """
        if not await self.client.connect():
            return False

        if self.tools:
            await self.client.add_tools(self.tools)

        self.client.start_audio_input()
        self.client.start_audio_output()

        self.is_active = True
        return True

    async def run(self) -> None:
        """Run the conversation loop (blocking).

        This will run until stop() is called or connection is lost.
        """
        if not self.is_active:
            if not await self.start():
                return

        await self.client.listen_events()

    async def stop(self) -> None:
        """Stop the conversation loop."""
        self.is_active = False
        await self.client.disconnect()


# Aria tool definitions for Realtime API
# These use vision-guided execution for reliable action completion
ARIA_REALTIME_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "name": "execute_task",
        "description": "Execute a high-level task using vision to plan and verify. This is the PREFERRED tool for most actions. Describe what you want to accomplish and the system will use vision to figure out how to do it. Examples: 'open a new Chrome window', 'click the File menu and select New', 'scroll down on this page'",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Description of what to accomplish (e.g., 'open a new Chrome window', 'click the submit button')"
                }
            },
            "required": ["task"]
        }
    },
    {
        "type": "function",
        "name": "click",
        "description": "Click on a UI element by describing what to click. The system will use vision to find the element and click it. Do NOT guess coordinates - describe the target instead.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Description of what to click (e.g., 'the File menu', 'the blue Submit button', 'the search box')"
                }
            },
            "required": ["target"]
        }
    },
    {
        "type": "function",
        "name": "open_menu_item",
        "description": "Open a menu and click a menu item. Uses vision to find and click accurately.",
        "parameters": {
            "type": "object",
            "properties": {
                "menu": {
                    "type": "string",
                    "description": "The menu to open (e.g., 'File', 'Edit', 'Chrome')"
                },
                "item": {
                    "type": "string",
                    "description": "The menu item to click (e.g., 'New Window', 'Copy', 'Preferences')"
                }
            },
            "required": ["menu", "item"]
        }
    },
    {
        "type": "function",
        "name": "type_text",
        "description": "Type text at the current cursor position. Uses clipboard paste for reliability.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to type"}
            },
            "required": ["text"]
        }
    },
    {
        "type": "function",
        "name": "press_key",
        "description": "Press a single key (enter, tab, escape, space, backspace, delete, up, down, left, right, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key to press"}
            },
            "required": ["key"]
        }
    },
    {
        "type": "function",
        "name": "hotkey",
        "description": "Press a keyboard shortcut. Example: ['command', 'c'] for Cmd+C",
        "parameters": {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keys to press together"
                }
            },
            "required": ["keys"]
        }
    },
    {
        "type": "function",
        "name": "open_app",
        "description": "Open an application by name (e.g., 'Safari', 'Terminal', 'VS Code')",
        "parameters": {
            "type": "object",
            "properties": {
                "app": {"type": "string", "description": "Application name"}
            },
            "required": ["app"]
        }
    },
    {
        "type": "function",
        "name": "open_url",
        "description": "Open a URL in the default browser.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to open"}
            },
            "required": ["url"]
        }
    },
    {
        "type": "function",
        "name": "scroll",
        "description": "Scroll the mouse wheel. Positive = up, negative = down.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "integer",
                    "description": "Scroll amount (positive=up, negative=down)"
                }
            },
            "required": ["amount"]
        }
    },
    {
        "type": "function",
        "name": "remember",
        "description": "Store information in long-term memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "The fact to remember"},
                "category": {
                    "type": "string",
                    "enum": ["preference", "personal", "work", "habit", "project", "other"],
                    "description": "Category for the memory"
                }
            },
            "required": ["fact"]
        }
    },
    {
        "type": "function",
        "name": "recall",
        "description": "Search long-term memory for relevant information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "n_results": {"type": "integer", "description": "Number of results to return"}
            },
            "required": ["query"]
        }
    }
]


def create_aria_tool_handler(control_module, memory_module) -> Callable[[str, str, Dict], str]:
    """Create a tool handler that bridges Realtime API tools to Aria's control/memory modules.

    Args:
        control_module: The aria.control module or compatible interface.
        memory_module: The aria.memory module or compatible interface.

    Returns:
        A tool handler function for use with RealtimeConversationLoop.
    """
    def handle_tool(call_id: str, name: str, args: Dict) -> str:
        try:
            if name == "click":
                control_module.click(args["x"], args["y"])
                return json.dumps({"success": True})

            elif name == "double_click":
                control_module.double_click(args["x"], args["y"])
                return json.dumps({"success": True})

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
                memory_module.add(
                    args["fact"],
                    category=args.get("category", "other")
                )
                return json.dumps({"success": True, "message": "Remembered"})

            elif name == "recall":
                results = memory_module.search(
                    args["query"],
                    n_results=args.get("n_results", 5)
                )
                return json.dumps({"results": results})

            else:
                return json.dumps({"error": f"Unknown tool: {name}"})

        except Exception as e:
            return json.dumps({"error": str(e)})

    return handle_tool
