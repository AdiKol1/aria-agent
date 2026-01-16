"""
Gesture Recognition for Aria using MediaPipe.

Provides hands-free control:
- Thumbs up = Confirm
- Thumbs down = Cancel
- Open palm = Stop
- Pointing = Select
"""

import cv2
import threading
import time
from pathlib import Path
from typing import Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import urllib.request

# MediaPipe imports
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


class Gesture(Enum):
    """Recognized gestures."""
    NONE = "None"
    THUMBS_UP = "Thumb_Up"
    THUMBS_DOWN = "Thumb_Down"
    OPEN_PALM = "Open_Palm"
    CLOSED_FIST = "Closed_Fist"
    POINTING_UP = "Pointing_Up"
    VICTORY = "Victory"
    I_LOVE_YOU = "ILoveYou"


@dataclass
class GestureEvent:
    """A detected gesture event."""
    gesture: Gesture
    confidence: float
    handedness: str  # "Left" or "Right"
    timestamp: float


class GestureAction(Enum):
    """Actions triggered by gestures."""
    CONFIRM = "confirm"
    CANCEL = "cancel"
    STOP = "stop"
    PAUSE = "pause"
    SELECT = "select"
    NONE = "none"


# Gesture to action mapping
GESTURE_ACTION_MAP = {
    Gesture.THUMBS_UP: GestureAction.CONFIRM,
    Gesture.THUMBS_DOWN: GestureAction.CANCEL,
    Gesture.OPEN_PALM: GestureAction.STOP,
    Gesture.CLOSED_FIST: GestureAction.PAUSE,
    Gesture.POINTING_UP: GestureAction.SELECT,
}


class GestureRecognizer:
    """MediaPipe-based gesture recognition."""

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
    MODEL_PATH = Path.home() / ".aria" / "models" / "gesture_recognizer.task"

    def __init__(self, min_confidence: float = 0.7):
        """
        Initialize gesture recognizer.

        Args:
            min_confidence: Minimum confidence threshold for gesture detection (0.0-1.0)
        """
        self.min_confidence = min_confidence
        self.model_path = self._ensure_model()

        # Recognition state
        self.current_gesture: Optional[GestureEvent] = None
        self.callbacks: List[Callable[[GestureEvent], None]] = []

        # Camera and thread
        self.camera: Optional[cv2.VideoCapture] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Gesture stability (require gesture to persist)
        self.gesture_buffer: List[Gesture] = []
        self.buffer_size = 5
        self.last_stable_gesture = Gesture.NONE
        self.gesture_cooldown = 1.0  # seconds between triggers
        self.last_trigger_time = 0.0

        # Create recognizer
        self.recognizer: Optional[mp_vision.GestureRecognizer] = None
        self._create_recognizer()

    def _ensure_model(self) -> Path:
        """Download model if not present."""
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        if not self.MODEL_PATH.exists():
            print(f"Downloading gesture recognition model...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print(f"Model downloaded to {self.MODEL_PATH}")

        return self.MODEL_PATH

    def _create_recognizer(self):
        """Create the MediaPipe gesture recognizer."""
        base_options = mp_python.BaseOptions(
            model_asset_path=str(self.model_path)
        )

        options = mp_vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._on_result
        )

        self.recognizer = mp_vision.GestureRecognizer.create_from_options(options)

    def _on_result(self, result, output_image, timestamp_ms: int):
        """Callback for gesture recognition results."""
        if not result.gestures or not result.gestures[0]:
            self.gesture_buffer.append(Gesture.NONE)
            self._update_buffer()
            return

        gesture_result = result.gestures[0][0]
        handedness_result = result.handedness[0][0]

        try:
            gesture = Gesture(gesture_result.category_name)
        except ValueError:
            gesture = Gesture.NONE

        confidence = gesture_result.score

        # Add to buffer for stability
        if confidence >= self.min_confidence:
            self.gesture_buffer.append(gesture)
        else:
            self.gesture_buffer.append(Gesture.NONE)

        self._update_buffer()

        # Create event if stable gesture detected
        if gesture != Gesture.NONE and confidence >= self.min_confidence:
            self.current_gesture = GestureEvent(
                gesture=gesture,
                confidence=confidence,
                handedness=handedness_result.category_name,
                timestamp=time.time()
            )

    def _update_buffer(self):
        """Update gesture buffer and trigger if stable."""
        # Keep buffer at fixed size
        while len(self.gesture_buffer) > self.buffer_size:
            self.gesture_buffer.pop(0)

        if len(self.gesture_buffer) < self.buffer_size:
            return

        # Check if all recent gestures are the same
        if len(set(self.gesture_buffer)) == 1:
            stable_gesture = self.gesture_buffer[0]

            # Check cooldown
            now = time.time()
            if stable_gesture != Gesture.NONE and stable_gesture != self.last_stable_gesture:
                if now - self.last_trigger_time >= self.gesture_cooldown:
                    self._trigger_gesture(stable_gesture)
                    self.last_trigger_time = now

            self.last_stable_gesture = stable_gesture

    def _trigger_gesture(self, gesture: Gesture):
        """Trigger callbacks for a stable gesture."""
        if not self.current_gesture:
            return

        print(f"Gesture detected: {gesture.value} (confidence: {self.current_gesture.confidence:.2f})")

        for callback in self.callbacks:
            try:
                callback(self.current_gesture)
            except Exception as e:
                print(f"Gesture callback error: {e}")

    def on_gesture(self, callback: Callable[[GestureEvent], None]):
        """
        Register a gesture callback.

        Args:
            callback: Function to call when a gesture is detected.
                      Receives a GestureEvent with gesture, confidence, handedness, and timestamp.
        """
        self.callbacks.append(callback)

    def remove_callback(self, callback: Callable[[GestureEvent], None]):
        """
        Remove a previously registered callback.

        Args:
            callback: The callback function to remove.
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def get_action(self, gesture: Gesture) -> GestureAction:
        """
        Get the action associated with a gesture.

        Args:
            gesture: The detected gesture

        Returns:
            The corresponding GestureAction
        """
        return GESTURE_ACTION_MAP.get(gesture, GestureAction.NONE)

    def start(self):
        """Start gesture recognition in background thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self.thread.start()
        print("Gesture recognition started")

    def _recognition_loop(self):
        """Main recognition loop."""
        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            print("Failed to open camera for gesture recognition")
            self.running = False
            return

        frame_timestamp = 0

        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                continue

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create MediaPipe image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Process frame
            frame_timestamp += 33  # ~30fps
            if self.recognizer:
                self.recognizer.recognize_async(mp_image, frame_timestamp)

            # Small delay to limit CPU
            time.sleep(0.033)

        self.camera.release()

    def stop(self):
        """Stop gesture recognition."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.recognizer:
            self.recognizer.close()
        print("Gesture recognition stopped")

    def is_running(self) -> bool:
        """Check if gesture recognition is currently running."""
        return self.running

    def get_current_gesture(self) -> Optional[GestureEvent]:
        """Get the most recently detected gesture event."""
        return self.current_gesture

    def set_cooldown(self, seconds: float):
        """
        Set the cooldown period between gesture triggers.

        Args:
            seconds: Minimum time between gesture triggers
        """
        self.gesture_cooldown = seconds

    def set_buffer_size(self, size: int):
        """
        Set the buffer size for gesture stability detection.

        Args:
            size: Number of consecutive frames required for stable gesture
        """
        self.buffer_size = max(1, size)


class FacePresenceDetector:
    """Detect if user is present (looking at screen)."""

    def __init__(self, away_threshold: float = 5.0):
        """
        Initialize face presence detector.

        Args:
            away_threshold: Seconds without face detection before marking as away
        """
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        self.is_present = False
        self.last_seen: Optional[float] = None
        self.away_threshold = away_threshold
        self.callbacks: List[Callable[[bool], None]] = []

    def on_presence_change(self, callback: Callable[[bool], None]):
        """
        Register callback for presence changes.

        Args:
            callback: Function to call with True (present) or False (away)
        """
        self.callbacks.append(callback)

    def update(self, frame) -> bool:
        """
        Update presence status from camera frame.

        Args:
            frame: OpenCV BGR frame from camera

        Returns:
            Current presence status
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        was_present = self.is_present

        if results.detections:
            self.is_present = True
            self.last_seen = time.time()
        elif self.last_seen and (time.time() - self.last_seen) > self.away_threshold:
            self.is_present = False

        # Trigger callbacks on change
        if was_present != self.is_present:
            for callback in self.callbacks:
                try:
                    callback(self.is_present)
                except Exception as e:
                    print(f"Presence callback error: {e}")

        return self.is_present

    def close(self):
        """Release resources."""
        self.face_detection.close()


class GestureController:
    """
    High-level gesture controller for Aria integration.

    Combines gesture recognition with action handling for easy integration.
    """

    def __init__(
        self,
        on_confirm: Optional[Callable[[], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
        on_pause: Optional[Callable[[], None]] = None,
        on_select: Optional[Callable[[], None]] = None,
        auto_start: bool = False
    ):
        """
        Initialize gesture controller.

        Args:
            on_confirm: Callback for thumbs up gesture
            on_cancel: Callback for thumbs down gesture
            on_stop: Callback for open palm gesture
            on_pause: Callback for closed fist gesture
            on_select: Callback for pointing up gesture
            auto_start: Whether to start gesture recognition immediately
        """
        self.recognizer = GestureRecognizer()

        # Store action callbacks
        self.action_callbacks = {
            GestureAction.CONFIRM: on_confirm,
            GestureAction.CANCEL: on_cancel,
            GestureAction.STOP: on_stop,
            GestureAction.PAUSE: on_pause,
            GestureAction.SELECT: on_select,
        }

        # Register internal handler
        self.recognizer.on_gesture(self._handle_gesture)

        # Track state for confirmation workflows
        self.awaiting_confirmation = False
        self.confirmation_callback: Optional[Callable[[bool], None]] = None
        self.confirmation_timeout: float = 10.0
        self.confirmation_start_time: float = 0.0

        if auto_start:
            self.start()

    def _handle_gesture(self, event: GestureEvent):
        """Internal gesture handler that routes to action callbacks."""
        action = self.recognizer.get_action(event.gesture)

        # Handle confirmation workflow
        if self.awaiting_confirmation:
            if action == GestureAction.CONFIRM:
                self._complete_confirmation(True)
                return
            elif action in (GestureAction.CANCEL, GestureAction.STOP):
                self._complete_confirmation(False)
                return

        # Route to registered callback
        callback = self.action_callbacks.get(action)
        if callback:
            try:
                callback()
            except Exception as e:
                print(f"Action callback error for {action.value}: {e}")

    def _complete_confirmation(self, confirmed: bool):
        """Complete a pending confirmation."""
        self.awaiting_confirmation = False
        if self.confirmation_callback:
            try:
                self.confirmation_callback(confirmed)
            except Exception as e:
                print(f"Confirmation callback error: {e}")
            self.confirmation_callback = None

    def request_confirmation(
        self,
        callback: Callable[[bool], None],
        timeout: float = 10.0
    ):
        """
        Request gesture confirmation from user.

        Args:
            callback: Function to call with True (confirmed) or False (cancelled/timeout)
            timeout: Seconds to wait for gesture before timing out
        """
        self.awaiting_confirmation = True
        self.confirmation_callback = callback
        self.confirmation_timeout = timeout
        self.confirmation_start_time = time.time()

        # Start timeout checker
        def check_timeout():
            while self.awaiting_confirmation:
                time.sleep(0.5)
                if time.time() - self.confirmation_start_time > self.confirmation_timeout:
                    self._complete_confirmation(False)
                    break

        threading.Thread(target=check_timeout, daemon=True).start()

    def set_action_callback(self, action: GestureAction, callback: Optional[Callable[[], None]]):
        """
        Set or update a callback for a specific action.

        Args:
            action: The gesture action to handle
            callback: The callback function, or None to remove
        """
        self.action_callbacks[action] = callback

    def start(self):
        """Start gesture recognition."""
        self.recognizer.start()

    def stop(self):
        """Stop gesture recognition."""
        self.recognizer.stop()

    def is_running(self) -> bool:
        """Check if gesture recognition is running."""
        return self.recognizer.is_running()


# Singleton instances
_gesture_recognizer: Optional[GestureRecognizer] = None
_gesture_controller: Optional[GestureController] = None


def get_gesture_recognizer() -> GestureRecognizer:
    """Get the global gesture recognizer."""
    global _gesture_recognizer
    if _gesture_recognizer is None:
        _gesture_recognizer = GestureRecognizer()
    return _gesture_recognizer


def get_gesture_controller() -> GestureController:
    """Get the global gesture controller."""
    global _gesture_controller
    if _gesture_controller is None:
        _gesture_controller = GestureController()
    return _gesture_controller
