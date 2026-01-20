"""
Wake Word Detection for Aria

Uses Porcupine for wake word detection if available.
Falls back to Whisper-based detection using OpenAI API.
"""

import io
import struct
import threading
import time
import wave
from typing import Callable, Optional

import numpy as np

from .config import PORCUPINE_ACCESS_KEY, VOICE_SAMPLE_RATE, WAKE_WORD, OPENAI_API_KEY


class WakeWordDetector:
    """Detects wake word using Porcupine or Whisper-based fallback."""

    def __init__(self, on_wake: Callable[[], None]):
        """
        Initialize wake word detector.

        Args:
            on_wake: Callback function when wake word is detected
        """
        self.on_wake = on_wake
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._porcupine = None
        self._audio_stream = None
        self._use_porcupine = False
        self._use_whisper = False
        self._openai_client = None
        self._wake_word = WAKE_WORD.lower().strip()

        # Cooldown to prevent rapid re-triggers
        self._last_wake_time = 0
        self._cooldown_seconds = 3.0

        # Try to initialize Porcupine first
        if PORCUPINE_ACCESS_KEY:
            try:
                import pvporcupine
                # Note: "aria" is not a built-in keyword, would need custom .ppn
                self._porcupine = pvporcupine.create(
                    access_key=PORCUPINE_ACCESS_KEY,
                    keywords=["jarvis"],  # Placeholder - need custom for "aria"
                )
                self._use_porcupine = True
                print("Wake word: Using Porcupine (keyword: 'jarvis' as placeholder)")
            except Exception as e:
                print(f"Porcupine init failed: {e}")

        # Fall back to Whisper-based detection
        if not self._use_porcupine and OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=OPENAI_API_KEY)
                self._use_whisper = True
                print(f"Wake word: Using Whisper (listening for '{self._wake_word}')")
            except Exception as e:
                print(f"Whisper fallback init failed: {e}")

        if not self._use_porcupine and not self._use_whisper:
            print("Wake word: Disabled (use menubar ⌥ Space to activate)")

    def start(self):
        """Start listening for wake word."""
        if self.running:
            return

        self.running = True

        if self._use_porcupine:
            self._thread = threading.Thread(target=self._porcupine_loop, daemon=True)
            self._thread.start()
        elif self._use_whisper:
            self._thread = threading.Thread(target=self._whisper_loop, daemon=True)
            self._thread.start()
        else:
            # No background thread needed - user activates via menubar
            pass

    def stop(self):
        """Stop listening for wake word."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._porcupine:
            self._porcupine.delete()
            self._porcupine = None

    def _porcupine_loop(self):
        """Listen for wake word using Porcupine."""
        try:
            import pyaudio

            pa = pyaudio.PyAudio()
            stream = pa.open(
                rate=self._porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self._porcupine.frame_length,
            )

            print(f"Listening for wake word... (sample rate: {self._porcupine.sample_rate})")

            while self.running:
                pcm = stream.read(self._porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * self._porcupine.frame_length, pcm)

                keyword_index = self._porcupine.process(pcm)
                if keyword_index >= 0:
                    print("Wake word detected!")
                    self.on_wake()

            stream.stop_stream()
            stream.close()
            pa.terminate()

        except Exception as e:
            print(f"Porcupine loop error: {e}")
            self.running = False

    def _whisper_loop(self):
        """Listen for wake word using Whisper transcription.

        This is a fallback when Porcupine is not available.
        Continuously listens for short audio clips and checks for wake word.
        """
        try:
            import pyaudio

            pa = pyaudio.PyAudio()

            # Audio settings - match Realtime API for compatibility
            sample_rate = 24000  # Match Realtime API sample rate
            chunk_size = 1024
            channels = 1

            # We'll record 1.5 second clips and check for wake word
            clip_duration = 1.5
            frames_per_clip = int(sample_rate * clip_duration / chunk_size)

            # Voice activity detection threshold - lowered for better sensitivity
            vad_threshold = 0.005  # RMS energy threshold (lowered from 0.01)

            stream = pa.open(
                rate=sample_rate,
                channels=channels,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=chunk_size,
            )

            print(f"Listening for wake word '{self._wake_word}'...")

            frames = []
            speech_detected = False
            silence_frames = 0
            max_silence_frames = 8  # After 8 silent frames (~0.5s), process the clip

            while self.running:
                # Read audio chunk
                data = stream.read(chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                # Calculate RMS energy for VAD
                rms = np.sqrt(np.mean(audio_data ** 2))

                if rms > vad_threshold:
                    # Voice detected
                    speech_detected = True
                    silence_frames = 0
                    frames.append(data)
                elif speech_detected:
                    # Still collecting after speech
                    silence_frames += 1
                    frames.append(data)

                    # If enough silence after speech, process the clip
                    if silence_frames >= max_silence_frames:
                        # Check cooldown
                        if time.time() - self._last_wake_time < self._cooldown_seconds:
                            frames = []
                            speech_detected = False
                            silence_frames = 0
                            continue

                        # Convert frames to WAV
                        if len(frames) > 5:  # At least 0.3 seconds
                            audio_bytes = self._frames_to_wav(frames, sample_rate)

                            # Transcribe with Whisper
                            try:
                                transcript = self._transcribe(audio_bytes)
                                if transcript:
                                    transcript_lower = transcript.lower().strip()

                                    # Check if wake word is in transcript
                                    # Support both "aria" and "hey aria"
                                    if self._wake_word in transcript_lower or "aria" in transcript_lower:
                                        print(f"Wake word detected: '{transcript}'")
                                        self._last_wake_time = time.time()
                                        self.on_wake()
                            except Exception:
                                pass  # Ignore transcription errors

                        # Reset for next clip
                        frames = []
                        speech_detected = False
                        silence_frames = 0

                # Prevent frames from growing too large (max 5 seconds)
                if len(frames) > frames_per_clip * 3:
                    frames = frames[-frames_per_clip:]

            stream.stop_stream()
            stream.close()
            pa.terminate()

        except Exception as e:
            print(f"Whisper wake word loop error: {e}")
            import traceback
            traceback.print_exc()
            self.running = False

    def _frames_to_wav(self, frames: list, sample_rate: int) -> io.BytesIO:
        """Convert audio frames to WAV format for Whisper."""
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(sample_rate)
            wav.writeframes(b''.join(frames))
        buffer.seek(0)
        return buffer

    def _transcribe(self, audio_bytes: io.BytesIO) -> Optional[str]:
        """Transcribe audio using Whisper."""
        if not self._openai_client:
            return None

        try:
            transcript = self._openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=("wake.wav", audio_bytes, "audio/wav"),
                response_format="text",
                language="en"
            )
            return transcript.strip()
        except Exception:
            return None

    def _keyboard_loop(self):
        """Fallback: Not used - activation via menubar instead."""
        pass


class SimpleWakeWord:
    """
    Simple wake word detector without Porcupine.
    Uses voice activity detection and a simple pattern match.
    """

    def __init__(self, on_wake: Callable[[], None]):
        self.on_wake = on_wake
        self.running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the simple detector."""
        if self.running:
            return
        self.running = True
        print("Simple wake word: Press Ctrl+Space or say 'hey' loudly")
        # For v0.1, we'll use a simpler approach

    def stop(self):
        """Stop the detector."""
        self.running = False


# Factory function
def create_wake_detector(on_wake: Callable[[], None]) -> WakeWordDetector:
    """Create the appropriate wake word detector."""
    return WakeWordDetector(on_wake)
