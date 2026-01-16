"""
Voice Interface for Aria

Handles speech-to-text and text-to-speech using OpenAI.

Supports two modes:
1. Traditional mode: Whisper STT -> Claude -> TTS (default)
2. Realtime mode: OpenAI Realtime API for sub-second latency

Set REALTIME_VOICE_ENABLED=True in config to enable realtime mode.
"""

import asyncio
import io
import queue
import threading
import time
import wave
from typing import Callable, Dict, List, Optional

import numpy as np
import sounddevice as sd
from openai import OpenAI

from .config import (
    OPENAI_API_KEY,
    VOICE_SAMPLE_RATE,
    VOICE_CHANNELS,
    REALTIME_VOICE_ENABLED,
    REALTIME_VOICE_MODEL,
    REALTIME_VOICE_VOICE,
    REALTIME_VAD_THRESHOLD,
    REALTIME_SILENCE_DURATION_MS,
)


class VoiceInterface:
    """Handles voice input and output."""

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.is_listening = False
        self.is_speaking = False
        self._audio_queue: queue.Queue = queue.Queue()
        self._recording_thread: Optional[threading.Thread] = None

    def listen(self, timeout: float = 10.0, silence_threshold: float = 0.015) -> Optional[str]:
        """
        Listen for voice input and transcribe.

        Args:
            timeout: Maximum seconds to listen
            silence_threshold: RMS threshold for silence detection (higher = less sensitive)

        Returns:
            Transcribed text or None if nothing detected
        """
        print("Listening...")
        self.is_listening = True

        # Record audio with shorter silence duration for snappier response
        audio_data = self._record_until_silence(
            timeout,
            silence_threshold,
            silence_duration=0.7  # Shorter silence to end recording faster
        )

        self.is_listening = False

        if audio_data is None or len(audio_data) < VOICE_SAMPLE_RATE * 0.5:  # Less than 0.5 seconds
            print("No audio detected")
            return None

        # Convert to WAV for API
        wav_buffer = self._numpy_to_wav(audio_data)

        # Transcribe with Whisper
        try:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.wav", wav_buffer, "audio/wav"),
                response_format="text",
                language="en"  # Force English to avoid picking up foreign audio
            )
            text = transcript.strip()
            print(f"Heard: {text}")

            # Filter out Whisper hallucinations and garbage
            # Whisper hallucinates emojis, repeated chars, or foreign text from background noise
            if self._is_likely_hallucination(text):
                print("Filtered: Likely Whisper hallucination (background noise)")
                return None

            return text
        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    def _is_likely_hallucination(self, text: str) -> bool:
        """
        Detect Whisper hallucinations from background noise.

        Whisper often outputs garbage when processing:
        - Music
        - Foreign language audio (when set to English)
        - Background noise
        - Audio feedback loops

        Common hallucination patterns:
        - Repeated emojis
        - Repeated characters/words
        - Very short nonsense
        - Known hallucination phrases
        """
        if not text:
            return True

        # Check for excessive emojis (Whisper hallucinates these from music)
        emoji_count = sum(1 for c in text if ord(c) > 0x1F600)
        if emoji_count > 5:
            return True

        # Check for repeated character patterns (e.g., "😍😍😍😍😍")
        if len(text) > 10:
            # Check if more than 50% is the same character repeated
            from collections import Counter
            char_counts = Counter(text.replace(" ", ""))
            if char_counts and char_counts.most_common(1)[0][1] > len(text) * 0.5:
                return True

        # Known Whisper hallucination phrases (often from background noise)
        hallucination_phrases = [
            "thanks for watching",
            "subscribe",
            "like and subscribe",
            "see you next time",
            "please subscribe",
            "don't forget to subscribe",
            "hit the bell",
            "チャンネル登録",  # Japanese "channel subscription"
            "novo notebook",  # Portuguese tech review
            "music playing",
            "background music",
        ]
        text_lower = text.lower()
        for phrase in hallucination_phrases:
            if phrase.lower() in text_lower:
                return True

        # Very short text that's likely noise
        words = text.split()
        if len(words) <= 1 and len(text) < 5:
            return True

        return False

    def speak(self, text: str, voice: str = "nova") -> bool:
        """
        Speak text using OpenAI TTS.

        Args:
            text: Text to speak
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
                   nova is faster and more natural for conversation
        """
        if not text:
            return False

        # Truncate very long responses for voice (keep it conversational)
        # But cut at sentence boundaries for natural speech
        if len(text) > 800:
            # Try to cut at a sentence boundary
            truncated = text[:800]
            # Find last sentence ending
            for ending in ['. ', '! ', '? ']:
                last_pos = truncated.rfind(ending)
                if last_pos > 400:  # Don't cut too short
                    text = truncated[:last_pos + 1]
                    break
            else:
                # No good sentence break, just truncate
                text = truncated + "..."

        print(f"Speaking: {text[:50]}...")
        self.is_speaking = True

        try:
            # Generate speech with tts-1 (fast) - speed=1.0 for natural cadence
            print("  Calling OpenAI TTS...")
            response = self.client.audio.speech.create(
                model="tts-1",  # Fast model
                voice=voice,
                input=text,
                response_format="pcm",
                speed=1.0  # Natural speed (1.1 was causing choppiness)
            )
            print(f"  TTS response received ({len(response.content)} bytes)")

            # Play audio with proper buffering
            audio_data = np.frombuffer(response.content, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0

            print("  Playing audio...")
            # Use a larger blocksize for smoother playback
            sd.play(audio_float, samplerate=24000, blocksize=2048)
            sd.wait()
            print("  Audio playback complete")

            self.is_speaking = False
            return True

        except Exception as e:
            print(f"Speech error: {e}")
            import traceback
            traceback.print_exc()
            self.is_speaking = False
            return False

    def _record_until_silence(
        self,
        timeout: float,
        silence_threshold: float,
        silence_duration: float = 1.0
    ) -> Optional[np.ndarray]:
        """Record audio until silence is detected."""
        frames = []
        silence_frames = 0
        frames_per_second = VOICE_SAMPLE_RATE
        silence_frames_needed = int(silence_duration * frames_per_second / 1024)

        start_time = time.time()
        speech_started = False

        def callback(indata, frame_count, time_info, status):
            nonlocal silence_frames, speech_started
            if status:
                print(f"Audio status: {status}")

            # Calculate RMS
            rms = np.sqrt(np.mean(indata**2))

            if rms > silence_threshold:
                speech_started = True
                silence_frames = 0
            elif speech_started:
                silence_frames += 1

            frames.append(indata.copy())

        try:
            with sd.InputStream(
                samplerate=VOICE_SAMPLE_RATE,
                channels=VOICE_CHANNELS,
                dtype=np.float32,
                blocksize=1024,
                callback=callback
            ):
                while True:
                    time.sleep(0.1)

                    # Check timeout
                    if time.time() - start_time > timeout:
                        break

                    # Check for silence after speech
                    if speech_started and silence_frames >= silence_frames_needed:
                        break

            if not frames:
                return None

            return np.concatenate(frames)

        except Exception as e:
            print(f"Recording error: {e}")
            return None

    def _numpy_to_wav(self, audio: np.ndarray) -> io.BytesIO:
        """Convert numpy array to WAV format."""
        # Convert float32 to int16
        audio_int16 = (audio * 32767).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(VOICE_CHANNELS)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(VOICE_SAMPLE_RATE)
            wav.writeframes(audio_int16.tobytes())

        buffer.seek(0)
        return buffer


class ConversationLoop:
    """Manages a voice conversation with Aria."""

    def __init__(
        self,
        voice: VoiceInterface,
        on_user_speech: Callable[[str], str]
    ):
        """
        Initialize conversation loop.

        Args:
            voice: VoiceInterface instance
            on_user_speech: Callback that takes user text and returns response
        """
        self.voice = voice
        self.on_user_speech = on_user_speech
        self.active = False

    def start_turn(self):
        """Start a single conversation turn."""
        self.active = True

        # Play activation sound or say something
        self.voice.speak("Yes?")

        # Listen for user
        user_text = self.voice.listen(timeout=15.0)

        if user_text:
            # Get response
            response = self.on_user_speech(user_text)

            # Speak response
            if response:
                self.voice.speak(response)

        self.active = False

    def run_continuous(self, wake_detector):
        """Run continuous conversation with wake word."""
        print("Aria is ready. Say 'Hey Aria' to activate.")

        def on_wake():
            if not self.active:
                self.start_turn()

        wake_detector.on_wake = on_wake
        wake_detector.start()

        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            wake_detector.stop()


# Realtime Voice Support
# Import conditionally to avoid errors if websockets isn't installed

try:
    from .realtime_voice import (
        RealtimeVoiceClient,
        RealtimeConfig,
        RealtimeConversationLoop,
        RealtimeToolDefinition,
        ARIA_REALTIME_TOOLS,
        create_aria_tool_handler,
    )
    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    RealtimeVoiceClient = None
    RealtimeConfig = None
    RealtimeConversationLoop = None
    RealtimeToolDefinition = None
    ARIA_REALTIME_TOOLS = []
    create_aria_tool_handler = None


class RealtimeVoiceInterface:
    """Voice interface using OpenAI's Realtime API for sub-second latency.

    This provides a similar interface to VoiceInterface but uses the Realtime API
    for much faster voice-to-voice interactions.
    """

    def __init__(
        self,
        tools: Optional[List[Dict]] = None,
        tool_handler: Optional[Callable[[str, str, Dict], str]] = None,
        instructions: str = ""
    ):
        """Initialize the Realtime Voice Interface.

        Args:
            tools: List of tool definitions for function calling.
            tool_handler: Callback to handle tool calls.
            instructions: System instructions for the assistant.
        """
        if not REALTIME_AVAILABLE:
            raise ImportError(
                "Realtime voice is not available. "
                "Install websockets: pip install websockets"
            )

        self.config = RealtimeConfig(
            model=REALTIME_VOICE_MODEL,
            voice=REALTIME_VOICE_VOICE,
            sample_rate=VOICE_SAMPLE_RATE,
            vad_threshold=REALTIME_VAD_THRESHOLD,
            silence_duration_ms=REALTIME_SILENCE_DURATION_MS,
            instructions=instructions,
        )

        self.client = RealtimeVoiceClient(OPENAI_API_KEY, self.config)
        self.tools = tools or ARIA_REALTIME_TOOLS
        self.tool_handler = tool_handler

        self.is_connected = False
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._event_thread: Optional[threading.Thread] = None

        # Callbacks
        self.on_transcript: Optional[Callable[[str], None]] = None
        self.on_response: Optional[Callable[[str], None]] = None

    def _setup_callbacks(self) -> None:
        """Wire up internal callbacks to external ones."""
        if self.on_transcript:
            self.client.on_transcript = self.on_transcript
        if self.on_response:
            self.client.on_response_done = self.on_response
        if self.tool_handler:
            self.client.on_tool_call = self.tool_handler

    async def _async_connect(self) -> bool:
        """Async connect implementation."""
        if not await self.client.connect():
            return False

        if self.tools:
            await self.client.add_tools(self.tools)

        self.client.start_audio_input()
        self.client.start_audio_output()
        self.is_connected = True
        return True

    def connect(self) -> bool:
        """Connect to the Realtime API.

        Returns:
            True if connected successfully.
        """
        self._setup_callbacks()

        # Create a new event loop for the realtime connection
        self._event_loop = asyncio.new_event_loop()

        def run_event_loop():
            asyncio.set_event_loop(self._event_loop)
            self._event_loop.run_forever()

        self._event_thread = threading.Thread(target=run_event_loop, daemon=True)
        self._event_thread.start()

        # Connect in the event loop
        future = asyncio.run_coroutine_threadsafe(
            self._async_connect(),
            self._event_loop
        )

        try:
            return future.result(timeout=10.0)
        except Exception as e:
            print(f"[RealtimeVoice] Connection error: {e}")
            return False

    def start_listening(self) -> None:
        """Start the event listening loop.

        This runs the event loop that processes incoming audio and responses.
        Should be called after connect().
        """
        if not self.is_connected or not self._event_loop:
            return

        asyncio.run_coroutine_threadsafe(
            self.client.listen_events(),
            self._event_loop
        )

    def disconnect(self) -> None:
        """Disconnect from the Realtime API."""
        if self._event_loop and self.client:
            future = asyncio.run_coroutine_threadsafe(
                self.client.disconnect(),
                self._event_loop
            )
            try:
                future.result(timeout=5.0)
            except Exception:
                pass

        if self._event_loop:
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)
            self._event_loop = None

        self.is_connected = False

    def send_text(self, text: str) -> None:
        """Send a text message (will be responded to with audio).

        Args:
            text: The text to send.
        """
        if self._event_loop and self.is_connected:
            asyncio.run_coroutine_threadsafe(
                self.client.send_text(text),
                self._event_loop
            )

    def cancel_response(self) -> None:
        """Cancel the current response (for interruption)."""
        if self._event_loop and self.is_connected:
            asyncio.run_coroutine_threadsafe(
                self.client.cancel_response(),
                self._event_loop
            )

    @property
    def is_speaking(self) -> bool:
        """Check if the assistant is currently speaking."""
        return self.client.is_speaking if self.client else False

    @property
    def is_listening(self) -> bool:
        """Check if the client is currently listening for input."""
        return self.client.is_listening if self.client else False


class RealtimeConversationManager:
    """Manages a continuous conversation using the Realtime API.

    This is similar to ConversationLoop but designed for the always-on
    nature of the Realtime API.
    """

    def __init__(
        self,
        tools: Optional[List[Dict]] = None,
        tool_handler: Optional[Callable[[str, str, Dict], str]] = None,
        instructions: str = ""
    ):
        """Initialize the conversation manager.

        Args:
            tools: Tools to make available.
            tool_handler: Handler for tool calls.
            instructions: System instructions.
        """
        self.voice = RealtimeVoiceInterface(
            tools=tools,
            tool_handler=tool_handler,
            instructions=instructions
        )

        self.active = False
        self._transcripts: List[str] = []
        self._responses: List[str] = []

        # Wire up tracking callbacks
        self.voice.on_transcript = self._on_transcript
        self.voice.on_response = self._on_response

        # External callbacks
        self.on_user_speech: Optional[Callable[[str], None]] = None
        self.on_assistant_speech: Optional[Callable[[str], None]] = None

    def _on_transcript(self, text: str) -> None:
        """Handle user transcript."""
        self._transcripts.append(text)
        if self.on_user_speech:
            self.on_user_speech(text)

    def _on_response(self, text: str) -> None:
        """Handle assistant response."""
        self._responses.append(text)
        if self.on_assistant_speech:
            self.on_assistant_speech(text)

    def start(self) -> bool:
        """Start the conversation.

        Returns:
            True if started successfully.
        """
        if not self.voice.connect():
            return False

        self.voice.start_listening()
        self.active = True
        print("[RealtimeConversation] Started - listening for voice input")
        return True

    def stop(self) -> None:
        """Stop the conversation."""
        self.active = False
        self.voice.disconnect()
        print("[RealtimeConversation] Stopped")

    def run_blocking(self) -> None:
        """Run the conversation loop (blocks until interrupted).

        Press Ctrl+C to stop.
        """
        if not self.start():
            print("[RealtimeConversation] Failed to start")
            return

        try:
            print("[RealtimeConversation] Running... Press Ctrl+C to stop")
            while self.active:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[RealtimeConversation] Interrupted")
        finally:
            self.stop()


# Singletons
_voice: Optional[VoiceInterface] = None
_realtime_voice: Optional[RealtimeVoiceInterface] = None


def get_voice() -> VoiceInterface:
    """Get the singleton VoiceInterface instance (traditional mode)."""
    global _voice
    if _voice is None:
        _voice = VoiceInterface()
    return _voice


def get_realtime_voice(
    tools: Optional[List[Dict]] = None,
    tool_handler: Optional[Callable[[str, str, Dict], str]] = None,
    instructions: str = ""
) -> Optional[RealtimeVoiceInterface]:
    """Get the singleton RealtimeVoiceInterface instance.

    Args:
        tools: Tools for function calling (only used on first call).
        tool_handler: Handler for tool calls (only used on first call).
        instructions: System instructions (only used on first call).

    Returns:
        RealtimeVoiceInterface instance, or None if not available.
    """
    global _realtime_voice

    if not REALTIME_VOICE_ENABLED:
        print("[Voice] Realtime voice not enabled (OPENAI_API_KEY not set)")
        return None

    if not REALTIME_AVAILABLE:
        print("[Voice] Realtime voice not available (websockets not installed)")
        return None

    if _realtime_voice is None:
        _realtime_voice = RealtimeVoiceInterface(
            tools=tools,
            tool_handler=tool_handler,
            instructions=instructions
        )

    return _realtime_voice


def is_realtime_available() -> bool:
    """Check if realtime voice mode is available.

    Returns:
        True if realtime mode can be used.
    """
    return REALTIME_VOICE_ENABLED and REALTIME_AVAILABLE
