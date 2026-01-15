"""
Voice Interface for Aria

Handles speech-to-text and text-to-speech using OpenAI.
"""

import io
import queue
import threading
import time
import wave
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from openai import OpenAI

from .config import OPENAI_API_KEY, VOICE_SAMPLE_RATE, VOICE_CHANNELS


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
                response_format="text"
            )
            print(f"Heard: {transcript}")
            return transcript.strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return None

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
        if len(text) > 500:
            text = text[:500] + "..."

        print(f"Speaking: {text[:50]}...")
        self.is_speaking = True

        try:
            # Generate speech with tts-1 (faster) or tts-1-hd (better quality)
            response = self.client.audio.speech.create(
                model="tts-1",  # Faster model
                voice=voice,
                input=text,
                response_format="pcm",
                speed=1.1  # Slightly faster speech
            )

            # Play audio
            audio_data = np.frombuffer(response.content, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0

            sd.play(audio_float, samplerate=24000)
            sd.wait()

            self.is_speaking = False
            return True

        except Exception as e:
            print(f"Speech error: {e}")
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


# Singleton
_voice: Optional[VoiceInterface] = None


def get_voice() -> VoiceInterface:
    """Get the singleton VoiceInterface instance."""
    global _voice
    if _voice is None:
        _voice = VoiceInterface()
    return _voice
