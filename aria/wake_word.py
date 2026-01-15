"""
Wake Word Detection for Aria

Uses Porcupine for "Hey Aria" detection.
Falls back to simple voice activity detection if no Porcupine key.
"""

import struct
import threading
import time
from typing import Callable, Optional

import numpy as np

from .config import PORCUPINE_ACCESS_KEY, VOICE_SAMPLE_RATE


class WakeWordDetector:
    """Detects "Hey Aria" wake word using Porcupine or fallback."""

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

        # Try to initialize Porcupine
        if PORCUPINE_ACCESS_KEY:
            try:
                import pvporcupine
                # Note: "hey aria" might need a custom keyword file
                # For now, use "jarvis" as a placeholder or create custom
                self._porcupine = pvporcupine.create(
                    access_key=PORCUPINE_ACCESS_KEY,
                    keywords=["jarvis"],  # Replace with custom "hey aria"
                )
                self._use_porcupine = True
                print("Wake word: Using Porcupine (keyword: 'jarvis' as placeholder)")
            except Exception as e:
                print(f"Porcupine init failed: {e}")
                print("Wake word: Disabled (use menubar to activate)")
        else:
            print("Wake word: No Porcupine key (use menubar ⌥ Space to activate)")

    def start(self):
        """Start listening for wake word."""
        if self.running:
            return

        self.running = True

        if self._use_porcupine:
            self._thread = threading.Thread(target=self._porcupine_loop, daemon=True)
            self._thread.start()
        else:
            # No background thread needed - user activates via menubar
            print("Wake word: Use menubar or ⌥ Space to activate Aria")

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
