"""
Qwen Omni + Claude Hybrid Voice System for Aria

This module provides a cost-effective voice interface that:
- Uses Qwen2.5-Omni locally for voice I/O (FREE)
- Routes complex tasks to Claude API (pay per use)
- Handles simple commands entirely locally

Architecture:
    Audio In → Qwen STT → Router → [Local/Claude] → Qwen TTS → Audio Out
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

try:
    import numpy as np
except ImportError:
    np = None  # Will be imported lazily when needed


@dataclass
class QwenVoiceConfig:
    """Configuration for Qwen hybrid voice system."""

    # Model settings
    qwen_model: str = "Qwen/Qwen2.5-Omni-7B"
    qwen_quantization: str = "4bit"  # 4bit, 8bit, or none

    # Claude settings (for complex tasks)
    claude_model: str = "claude-sonnet-4-20250514"
    claude_model_complex: str = "claude-opus-4-5-20251101"

    # Audio settings
    sample_rate: int = 16000
    chunk_duration_ms: int = 100

    # Routing settings
    use_local_for_simple: bool = True
    complexity_threshold: float = 0.7

    # TTS settings
    tts_voice: str = "default"
    tts_speed: float = 1.0


# Simple command patterns that can be handled locally
SIMPLE_COMMAND_PATTERNS = [
    # App control
    (r"^(?:open|launch|start)\s+(.+)$", "open_app"),
    (r"^(?:close|quit|exit)\s+(.+)$", "close_app"),

    # Navigation
    (r"^(?:go to|navigate to|open)\s+(https?://\S+|\S+\.\S+)$", "open_url"),
    (r"^scroll\s+(up|down)(?:\s+(\d+))?$", "scroll"),

    # Typing
    (r"^type\s+[\"']?(.+?)[\"']?$", "type_text"),
    (r"^press\s+(.+)$", "press_key"),

    # Shortcuts
    (r"^(?:do\s+)?(?:cmd|command)\s*\+\s*(\w)$", "hotkey"),
    (r"^copy$", "copy"),
    (r"^paste$", "paste"),
    (r"^undo$", "undo"),
    (r"^save$", "save"),
    (r"^new tab$", "new_tab"),
    (r"^close tab$", "close_tab"),

    # Simple queries
    (r"^what(?:'s| is) the time\??$", "get_time"),
    (r"^what(?:'s| is) the date\??$", "get_date"),
]

# App name mappings
APP_ALIASES = {
    "chrome": "Google Chrome",
    "safari": "Safari",
    "finder": "Finder",
    "terminal": "Terminal",
    "code": "Visual Studio Code",
    "vscode": "Visual Studio Code",
    "vs code": "Visual Studio Code",
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


class QwenOmniModel:
    """
    Wrapper for Qwen2.5-Omni model.

    Handles:
    - Speech-to-text (audio → text)
    - Text generation (text → text)
    - Text-to-speech (text → audio)
    """

    def __init__(self, config: QwenVoiceConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._is_loaded = False
        self._load_lock = threading.Lock()

    def _ensure_loaded(self):
        """Lazy load the model on first use."""
        if self._is_loaded:
            return

        with self._load_lock:
            if self._is_loaded:
                return

            print("[QwenOmni] Loading model (this may take a minute)...")
            start = time.time()

            try:
                from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
                import torch

                # Determine device and dtype
                if torch.backends.mps.is_available():
                    device = "mps"
                    dtype = torch.float16
                elif torch.cuda.is_available():
                    device = "cuda"
                    dtype = torch.bfloat16
                else:
                    device = "cpu"
                    dtype = torch.float32

                # Load with quantization if specified
                load_kwargs = {
                    "torch_dtype": dtype,
                    "device_map": "auto",
                }

                if self.config.qwen_quantization == "4bit":
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=dtype,
                    )
                elif self.config.qwen_quantization == "8bit":
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )

                self._processor = Qwen2_5OmniProcessor.from_pretrained(self.config.qwen_model)
                self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    self.config.qwen_model,
                    **load_kwargs
                )

                self._is_loaded = True
                print(f"[QwenOmni] Model loaded in {time.time() - start:.1f}s on {device}")

            except ImportError as e:
                print(f"[QwenOmni] Failed to load: {e}")
                print("[QwenOmni] Falling back to Ollama...")
                self._use_ollama_fallback = True
                self._is_loaded = True

    def transcribe(self, audio: "np.ndarray") -> str:
        """Convert audio to text using Qwen Omni's STT."""
        self._ensure_loaded()

        if hasattr(self, '_use_ollama_fallback') and self._use_ollama_fallback:
            return self._transcribe_with_whisper(audio)

        try:
            import torch

            # Prepare audio input
            inputs = self._processor(
                audios=audio,
                sampling_rate=self.config.sample_rate,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            # Generate transcription
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )

            text = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            return text.strip()

        except Exception as e:
            print(f"[QwenOmni] Transcription error: {e}")
            return self._transcribe_with_whisper(audio)

    def _transcribe_with_whisper(self, audio: "np.ndarray") -> str:
        """Fallback to mlx-whisper for transcription."""
        try:
            import mlx_whisper
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo="mlx-community/whisper-base-mlx",
            )
            return result.get("text", "").strip()
        except ImportError:
            # Final fallback: save to file and use whisper CLI
            import tempfile
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, self.config.sample_rate)
                result = subprocess.run(
                    ["whisper", f.name, "--model", "base", "--output_format", "txt"],
                    capture_output=True,
                    text=True,
                )
                os.unlink(f.name)
                return result.stdout.strip()

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text response using Qwen Omni."""
        self._ensure_loaded()

        if hasattr(self, '_use_ollama_fallback') and self._use_ollama_fallback:
            return self._generate_with_ollama(prompt, system_prompt)

        try:
            import torch

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=text, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                )

            response = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            # Extract just the assistant's response
            if "assistant" in response.lower():
                response = response.split("assistant")[-1].strip()
            return response

        except Exception as e:
            print(f"[QwenOmni] Generation error: {e}")
            return self._generate_with_ollama(prompt, system_prompt)

    def _generate_with_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Fallback to Ollama for text generation."""
        try:
            import requests

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "qwen2.5:7b",  # or deepseek-r1:7b
                    "messages": messages,
                    "stream": False,
                },
                timeout=60,
            )

            if response.ok:
                return response.json().get("message", {}).get("content", "")
            else:
                return "I'm having trouble processing that request."

        except Exception as e:
            print(f"[Ollama] Error: {e}")
            return "I'm having trouble processing that request."

    def speak(self, text: str) -> "np.ndarray":
        """Convert text to speech using Qwen Omni's TTS."""
        self._ensure_loaded()

        if hasattr(self, '_use_ollama_fallback') and self._use_ollama_fallback:
            return self._speak_with_system_tts(text)

        try:
            import torch

            # Prepare TTS input
            inputs = self._processor(
                text=text,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            # Generate audio
            with torch.no_grad():
                audio_output = self._model.generate_speech(**inputs)

            return audio_output.cpu().numpy()

        except Exception as e:
            print(f"[QwenOmni] TTS error: {e}")
            return self._speak_with_system_tts(text)

    def _speak_with_system_tts(self, text: str) -> "Optional[np.ndarray]":
        """Fallback to system TTS (macOS say command)."""
        # For now, just use say command directly (non-blocking)
        subprocess.Popen(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return None  # Audio played directly


class TaskRouter:
    """
    Routes tasks to local execution or Claude API based on complexity.

    Simple tasks (open app, scroll, type) → Local execution
    Complex tasks (reasoning, analysis) → Claude API
    """

    def __init__(self, config: QwenVoiceConfig):
        self.config = config
        self._patterns = [(re.compile(p, re.IGNORECASE), action) for p, action in SIMPLE_COMMAND_PATTERNS]

    def classify(self, text: str) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        Classify a command as simple or complex.

        Returns:
            (route, params) where route is "local" or "claude"
            params contains extracted parameters for local commands
        """
        text = text.strip().lower()

        # Check against simple patterns
        for pattern, action in self._patterns:
            match = pattern.match(text)
            if match:
                params = {"action": action, "groups": match.groups()}
                return ("local", params)

        # Check for question words that need reasoning
        question_words = ["why", "how", "explain", "what do you think", "analyze", "help me"]
        if any(text.startswith(w) or w in text for w in question_words):
            return ("claude", None)

        # Check for multi-step indicators
        multi_step_indicators = ["and then", "after that", "first", "next", "finally"]
        if any(ind in text for ind in multi_step_indicators):
            return ("claude", None)

        # Default to Claude for anything uncertain
        if len(text.split()) > 10:
            return ("claude", None)

        # Short, unrecognized commands - try local first
        return ("local", {"action": "unknown", "text": text})

    def is_simple(self, text: str) -> bool:
        """Check if a command can be handled locally."""
        route, _ = self.classify(text)
        return route == "local"


class QwenClaudeHybrid:
    """
    Main hybrid voice system combining Qwen Omni and Claude.

    Usage:
        hybrid = QwenClaudeHybrid()
        await hybrid.start()

        # Process voice input
        response = await hybrid.process_audio(audio_data)
    """

    def __init__(self, config: Optional[QwenVoiceConfig] = None):
        self.config = config or QwenVoiceConfig()

        # Components
        self.qwen = QwenOmniModel(self.config)
        self.router = TaskRouter(self.config)
        self._claude_client = None

        # Control module
        self._control = None

        # State
        self.is_running = False
        self._audio_queue = queue.Queue()

        # Callbacks
        self.on_transcription: Optional[Callable[[str], None]] = None
        self.on_response: Optional[Callable[[str], None]] = None
        self.on_action: Optional[Callable[[str], None]] = None

    def _get_control(self):
        """Lazy load control module."""
        if self._control is None:
            from .control import get_control
            self._control = get_control()
        return self._control

    def _get_claude(self):
        """Lazy load Claude client."""
        if self._claude_client is None:
            from anthropic import Anthropic
            from .config import ANTHROPIC_API_KEY
            self._claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        return self._claude_client

    async def process_text(self, text: str) -> str:
        """
        Process a text command and return response.

        Args:
            text: User's command/query

        Returns:
            Response text
        """
        if self.on_transcription:
            self.on_transcription(text)

        # Route the command
        route, params = self.router.classify(text)

        if route == "local" and self.config.use_local_for_simple:
            response = await self._handle_local(text, params)
        else:
            response = await self._handle_claude(text)

        if self.on_response:
            self.on_response(response)

        return response

    async def process_audio(self, audio: "np.ndarray") -> str:
        """
        Process audio input and return response.

        Args:
            audio: Audio data as numpy array

        Returns:
            Response text (also spoken via TTS)
        """
        # Transcribe
        text = self.qwen.transcribe(audio)
        print(f"[User]: {text}")

        if not text or len(text.strip()) < 2:
            return ""

        # Process
        response = await self.process_text(text)

        # Speak response
        if response:
            print(f"[Aria]: {response}")
            self.qwen.speak(response)

        return response

    async def _handle_local(self, text: str, params: Optional[Dict]) -> str:
        """Handle a simple command locally."""
        control = self._get_control()

        if not params:
            # Unknown command - use local LLM
            return self.qwen.generate(
                text,
                system_prompt="You are Aria, a helpful voice assistant. Be brief and conversational."
            )

        action = params.get("action")
        groups = params.get("groups", ())

        try:
            if action == "open_app":
                app_name = groups[0] if groups else ""
                app_name = APP_ALIASES.get(app_name.lower(), app_name.title())
                control.open_app(app_name)
                if self.on_action:
                    self.on_action(f"open_app: {app_name}")
                return f"Opening {app_name}."

            elif action == "close_app":
                app_name = groups[0] if groups else ""
                app_name = APP_ALIASES.get(app_name.lower(), app_name.title())
                control.quit_app(app_name)
                if self.on_action:
                    self.on_action(f"close_app: {app_name}")
                return f"Closing {app_name}."

            elif action == "open_url":
                url = groups[0] if groups else ""
                if not url.startswith("http"):
                    url = "https://" + url
                control.open_url(url)
                if self.on_action:
                    self.on_action(f"open_url: {url}")
                return f"Opening {url}."

            elif action == "scroll":
                direction = groups[0] if groups else "down"
                amount = int(groups[1]) if len(groups) > 1 and groups[1] else 3
                scroll_amount = amount if direction == "up" else -amount
                control.scroll(scroll_amount)
                if self.on_action:
                    self.on_action(f"scroll: {direction} {amount}")
                return f"Scrolled {direction}."

            elif action == "type_text":
                text_to_type = groups[0] if groups else ""
                control.type_text(text_to_type)
                if self.on_action:
                    self.on_action(f"type: {text_to_type[:20]}...")
                return "Typed it."

            elif action == "press_key":
                key = groups[0] if groups else ""
                control.press_key(key.lower())
                if self.on_action:
                    self.on_action(f"press: {key}")
                return f"Pressed {key}."

            elif action == "hotkey":
                key = groups[0] if groups else ""
                control.hotkey("command", key.lower())
                if self.on_action:
                    self.on_action(f"hotkey: cmd+{key}")
                return f"Done."

            elif action == "copy":
                control.copy()
                return "Copied."

            elif action == "paste":
                control.paste()
                return "Pasted."

            elif action == "undo":
                control.undo()
                return "Undone."

            elif action == "save":
                control.save()
                return "Saved."

            elif action == "new_tab":
                control.new_tab()
                return "New tab opened."

            elif action == "close_tab":
                control.close_tab()
                return "Tab closed."

            elif action == "get_time":
                from datetime import datetime
                now = datetime.now()
                return f"It's {now.strftime('%I:%M %p')}."

            elif action == "get_date":
                from datetime import datetime
                now = datetime.now()
                return f"It's {now.strftime('%A, %B %d, %Y')}."

            else:
                # Unknown action - use local LLM for simple response
                return self.qwen.generate(
                    text,
                    system_prompt="You are Aria, a helpful voice assistant. Be brief and conversational."
                )

        except Exception as e:
            print(f"[Local] Error: {e}")
            return f"Sorry, I couldn't do that: {str(e)}"

    async def _handle_claude(self, text: str) -> str:
        """Handle a complex command via Claude API."""
        try:
            claude = self._get_claude()

            # Determine which model to use based on complexity
            model = self.config.claude_model
            if any(word in text.lower() for word in ["analyze", "explain", "complex", "detailed"]):
                model = self.config.claude_model_complex

            response = claude.messages.create(
                model=model,
                max_tokens=500,
                system="""You are Aria, a helpful voice assistant that controls a Mac computer.
Be conversational and concise - your responses will be spoken aloud.
If the user wants you to do something on their computer, describe what you would do.
Keep responses under 2-3 sentences for simple queries.""",
                messages=[{"role": "user", "content": text}]
            )

            return response.content[0].text

        except Exception as e:
            print(f"[Claude] Error: {e}")
            # Fall back to local model
            return self.qwen.generate(
                text,
                system_prompt="You are Aria, a helpful voice assistant. Be brief and conversational."
            )

    async def start_listening(self):
        """Start the voice listening loop."""
        import sounddevice as sd

        self.is_running = True
        print("\n" + "=" * 50)
        print("QWEN + CLAUDE HYBRID VOICE SYSTEM")
        print("=" * 50)
        print("Qwen Omni: Voice I/O (FREE, local)")
        print("Claude: Complex reasoning (API)")
        print("=" * 50)
        print("Speak naturally. Press Ctrl+C to stop.")
        print("=" * 50 + "\n")

        # Audio buffer for collecting speech
        audio_buffer = []
        silence_threshold = 0.01
        silence_duration = 0
        max_silence = 0.8  # seconds
        is_speaking = False

        def audio_callback(indata, frames, time_info, status):
            nonlocal audio_buffer, silence_duration, is_speaking

            if not self.is_running:
                return

            # Calculate audio energy
            energy = np.abs(indata).mean()

            if energy > silence_threshold:
                is_speaking = True
                silence_duration = 0
                audio_buffer.append(indata.copy())
            elif is_speaking:
                silence_duration += frames / self.config.sample_rate
                audio_buffer.append(indata.copy())

                if silence_duration > max_silence:
                    # End of speech detected
                    if len(audio_buffer) > 5:  # Minimum length
                        audio = np.concatenate(audio_buffer, axis=0).flatten()
                        self._audio_queue.put(audio)

                    audio_buffer = []
                    silence_duration = 0
                    is_speaking = False

        # Start audio stream
        with sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=int(self.config.sample_rate * self.config.chunk_duration_ms / 1000),
            callback=audio_callback
        ):
            while self.is_running:
                try:
                    # Process any queued audio
                    audio = self._audio_queue.get(timeout=0.1)
                    await self.process_audio(audio)
                except queue.Empty:
                    pass
                except Exception as e:
                    print(f"[Error]: {e}")

    def stop(self):
        """Stop the voice system."""
        self.is_running = False
        print("\nStopping hybrid voice system...")


async def main():
    """Test the hybrid voice system."""
    hybrid = QwenClaudeHybrid()

    # Set up callbacks
    hybrid.on_transcription = lambda t: print(f"[Heard]: {t}")
    hybrid.on_response = lambda r: print(f"[Response]: {r}")
    hybrid.on_action = lambda a: print(f"[Action]: {a}")

    try:
        await hybrid.start_listening()
    except KeyboardInterrupt:
        hybrid.stop()


if __name__ == "__main__":
    asyncio.run(main())
