"""
Hybrid Voice Architecture

Gemini Live API handles voice I/O (instant STT + TTS)
Claude handles ALL reasoning and computer control

This gives:
- Instant voice response (Gemini's real-time streaming)
- Claude's full intelligence for complex tasks
- Computer use with verification loops
"""

import asyncio
import base64
import json
import os
import queue
import threading
import time
import re
import subprocess
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
import sounddevice as sd
from google import genai
from google.genai import types

from .config import GOOGLE_API_KEY, GEMINI_VOICE_VOICE
from .claude_computer_use import create_agent
from .control import get_control


@dataclass
class HybridConfig:
    """Configuration for hybrid voice system."""
    gemini_model: str = "gemini-2.0-flash-exp"
    voice: str = GEMINI_VOICE_VOICE or "Kore"
    sample_rate_in: int = 16000
    sample_rate_out: int = 24000


# Fast commands that execute instantly without Claude
FAST_COMMANDS = {
    r"^(?:open|launch|start)\s+(.+)$": "open_app",
    r"^(?:go to|navigate to|open)\s+((?:https?://)?[\w.-]+\.[\w]+.*)$": "open_url",
    r"^scroll\s+(up|down)(?:\s+(\d+))?$": "scroll",
    r"^type\s+(.+)$": "type_text",
    r"^press\s+(.+)$": "press_key",
}

APP_MAP = {
    "chrome": "Google Chrome", "safari": "Safari", "finder": "Finder",
    "terminal": "Terminal", "code": "Visual Studio Code", "vscode": "Visual Studio Code",
    "slack": "Slack", "spotify": "Spotify", "notes": "Notes", "calendar": "Calendar",
    "mail": "Mail", "messages": "Messages", "music": "Music", "photos": "Photos",
    "settings": "System Settings", "preferences": "System Settings",
}


class HybridVoiceSystem:
    """
    Hybrid voice system: Gemini for voice, Claude for brain.

    Gemini Live API provides:
    - Real-time speech-to-text (instant transcription)
    - Real-time text-to-speech (instant voice output)

    Claude provides:
    - Complex reasoning
    - Computer use (mouse, keyboard, screenshots)
    - Multi-step task execution
    - Verification loops
    """

    def __init__(self, config: Optional[HybridConfig] = None):
        self.config = config or HybridConfig()
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.control = get_control()

        # State
        self.is_running = False
        self.session = None
        self.current_transcript = ""
        self.is_processing = False
        self.last_response_time = 0

        # Audio
        self.audio_in_queue = queue.Queue()
        self.audio_out_queue = queue.Queue()

        # Claude agent (reusable)
        self._claude_agent = None

        # Callbacks
        self.on_user_speech: Optional[Callable[[str], None]] = None
        self.on_assistant_speech: Optional[Callable[[str], None]] = None
        self.on_action: Optional[Callable[[str], None]] = None

    def _get_gemini_config(self):
        """Get Gemini Live API configuration - voice only, no tools."""
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],  # Audio output only
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.config.voice
                    )
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part(text="""You are a voice output system. When you receive text, speak it naturally and concisely. Do NOT add your own commentary - just speak what you're given.""")]
            ),
        )

    def _try_fast_command(self, text: str) -> tuple[bool, str]:
        """Try to execute a simple command instantly."""
        text_lower = text.lower().strip().rstrip('.!?,')

        for pattern, action in FAST_COMMANDS.items():
            match = re.match(pattern, text_lower, re.IGNORECASE)
            if match:
                try:
                    if action == "open_app":
                        app_name = match.group(1).strip()
                        app = APP_MAP.get(app_name.lower(), app_name.title())
                        subprocess.run(["open", "-a", app], check=True,
                                      capture_output=True, timeout=5)
                        return True, f"Opened {app}."

                    elif action == "open_url":
                        url = match.group(1).strip()
                        if not url.startswith("http"):
                            url = "https://" + url
                        subprocess.run(["open", url], check=True,
                                      capture_output=True, timeout=5)
                        return True, f"Opened {url}."

                    elif action == "scroll":
                        direction = match.group(1)
                        amount = int(match.group(2)) if match.group(2) else 3
                        scroll_amount = amount if direction == "up" else -amount
                        self.control.scroll(scroll_amount)
                        return True, f"Scrolled {direction}."

                    elif action == "type_text":
                        text_to_type = match.group(1)
                        self.control.type_text(text_to_type)
                        return True, "Typed it."

                    elif action == "press_key":
                        key = match.group(1).strip().lower()
                        self.control.press_key(key)
                        return True, f"Pressed {key}."

                except Exception as e:
                    print(f"[Fast command error: {e}]")
                    return False, ""

        return False, ""

    def _run_claude_task(self, task: str) -> str:
        """Run a task using Claude Computer Use."""
        print(f"[Claude] Processing: {task}")

        def on_msg(msg):
            short = msg[:100] + "..." if len(msg) > 100 else msg
            print(f"[Claude]: {short}")

        def on_action(action):
            print(f"[Action]: {action}")
            if self.on_action:
                self.on_action(action)

        agent = create_agent(on_message=on_msg, on_action=on_action)
        result = agent.run(task)

        # Return short summary for voice
        if result:
            # First sentence or 200 chars
            first_sentence = result.split('.')[0] + '.'
            if len(first_sentence) > 200:
                return result[:200] + "..."
            return first_sentence
        return "Done."

    async def _process_user_input(self, transcript: str):
        """Process user input - fast path or Claude."""
        if self.is_processing:
            return

        self.is_processing = True
        transcript = transcript.strip()

        if not transcript or len(transcript) < 2:
            self.is_processing = False
            return

        print(f"\n[User]: {transcript}")
        if self.on_user_speech:
            self.on_user_speech(transcript)

        # Check for exit
        if transcript.lower() in ["quit", "exit", "stop", "bye", "goodbye"]:
            await self._send_text("Goodbye!")
            self.is_running = False
            self.is_processing = False
            return

        # Try fast command first
        handled, response = self._try_fast_command(transcript)
        if handled:
            print(f"[Fast]: {response}")
            await self._send_text(response)
            self.is_processing = False
            return

        # Fall back to Claude for complex tasks
        await self._send_text("Working on it...")

        # Run Claude in a thread to not block audio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._run_claude_task, transcript)

        print(f"[Result]: {result}")
        await self._send_text(result)

        if self.on_assistant_speech:
            self.on_assistant_speech(result)

        self.is_processing = False

    async def _send_text(self, text: str):
        """Send text to Gemini for TTS output."""
        if self.session:
            try:
                await self.session.send_client_content(
                    turns=types.Content(role="user", parts=[types.Part(text=text)]),
                    turn_complete=True
                )
            except Exception as e:
                print(f"[TTS Error]: {e}")

    async def _audio_input_loop(self):
        """Capture microphone audio and send to Gemini."""
        def audio_callback(indata, frames, time_info, status):
            if self.is_running and not self.is_processing:
                # Convert float32 to int16 PCM
                audio_int16 = (indata.flatten() * 32767).astype(np.int16)
                self.audio_in_queue.put(audio_int16.tobytes())

        try:
            with sd.InputStream(
                samplerate=self.config.sample_rate_in,
                channels=1,
                dtype=np.float32,
                blocksize=int(self.config.sample_rate_in * 0.1),  # 100ms chunks
                callback=audio_callback
            ):
                while self.is_running:
                    try:
                        audio_data = self.audio_in_queue.get(timeout=0.1)
                        if self.session and not self.is_processing:
                            await self.session.send_realtime_input(
                                media_chunks=[
                                    types.Blob(data=audio_data, mime_type="audio/pcm")
                                ]
                            )
                    except queue.Empty:
                        pass
                    except Exception as e:
                        if "closed" not in str(e).lower():
                            print(f"[Audio send error]: {e}")
        except Exception as e:
            print(f"[Audio input error]: {e}")

    async def _audio_output_loop(self):
        """Play audio from Gemini."""
        try:
            with sd.OutputStream(
                samplerate=self.config.sample_rate_out,
                channels=1,
                dtype=np.int16,
                blocksize=2048
            ) as stream:
                while self.is_running:
                    try:
                        audio_data = self.audio_out_queue.get(timeout=0.1)
                        if audio_data is not None:
                            audio_array = np.frombuffer(audio_data, dtype=np.int16)
                            stream.write(audio_array)
                    except queue.Empty:
                        pass
        except Exception as e:
            print(f"[Audio output error]: {e}")

    async def _receive_loop(self):
        """Receive responses from Gemini."""
        print("[DEBUG] Receive loop started", flush=True)
        try:
            async for response in self.session.receive():
                print(f"[DEBUG] Received response: {type(response).__name__}", flush=True)
                if not self.is_running:
                    break

                # Handle server content (transcripts, audio)
                if hasattr(response, 'server_content') and response.server_content:
                    content = response.server_content

                    # Check for turn completion
                    if hasattr(content, 'turn_complete') and content.turn_complete:
                        if self.current_transcript:
                            transcript = self.current_transcript.strip()
                            self.current_transcript = ""
                            if transcript and not self.is_processing:
                                asyncio.create_task(self._process_user_input(transcript))

                    # Handle model turn (audio output)
                    if hasattr(content, 'model_turn') and content.model_turn:
                        for part in content.model_turn.parts:
                            if hasattr(part, 'inline_data') and part.inline_data:
                                self.audio_out_queue.put(part.inline_data.data)

                # Handle tool calls - we don't use tools, just transcripts
                if hasattr(response, 'tool_call') and response.tool_call:
                    # Ignore tool calls - Gemini is voice only
                    pass

                # Handle input transcripts
                if hasattr(response, 'server_content'):
                    content = response.server_content
                    if hasattr(content, 'input_transcription') and content.input_transcription:
                        text = content.input_transcription.text
                        if text:
                            self.current_transcript = text

        except Exception as e:
            if "closed" not in str(e).lower():
                print(f"[Receive error]: {e}")

    async def run(self):
        """Run the hybrid voice system."""
        print("\n" + "="*50, flush=True)
        print("HYBRID VOICE SYSTEM", flush=True)
        print("="*50, flush=True)
        print("Gemini: Voice I/O (instant)", flush=True)
        print("Claude: Brain (complex tasks)", flush=True)
        print("="*50, flush=True)
        print("Speak naturally. Say 'quit' to exit.", flush=True)
        print("="*50 + "\n", flush=True)

        self.is_running = True

        print("[DEBUG] Connecting to Gemini Live API...", flush=True)
        try:
            async with self.client.aio.live.connect(
                model=self.config.gemini_model,
                config=self._get_gemini_config()
            ) as session:
                print("[DEBUG] Connected to Gemini!", flush=True)
                self.session = session

                # Start all loops
                print("[DEBUG] Starting audio loops...", flush=True)
                audio_in_task = asyncio.create_task(self._audio_input_loop())
                audio_out_task = asyncio.create_task(self._audio_output_loop())
                receive_task = asyncio.create_task(self._receive_loop())
                print("[DEBUG] Audio loops started", flush=True)

                # Send initial greeting
                print("[DEBUG] Sending Ready greeting...", flush=True)
                await self._send_text("Ready.")
                print("[DEBUG] Greeting sent, listening...", flush=True)

                # Wait for completion
                try:
                    await asyncio.gather(audio_in_task, audio_out_task, receive_task)
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            print(f"[Session error]: {e}")
        finally:
            self.is_running = False
            self.session = None
            print("\nHybrid system stopped.")


async def main():
    """Main entry point."""
    system = HybridVoiceSystem()

    system.on_user_speech = lambda t: None
    system.on_assistant_speech = lambda t: None
    system.on_action = lambda a: None

    try:
        await system.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        system.is_running = False


if __name__ == "__main__":
    asyncio.run(main())
