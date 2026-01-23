"""
Voice Mode Selector for Aria

This module provides a unified interface to switch between different voice backends:
1. OpenAI Realtime API (fast but expensive)
2. Qwen + Claude Hybrid (cost-effective)
3. Traditional pipeline (Whisper STT → Claude → TTS)

Usage:
    from aria.voice_mode import get_voice_system, VoiceMode

    # Use the hybrid system (recommended for cost savings)
    voice = get_voice_system(VoiceMode.HYBRID)
    await voice.start_listening()
"""

from enum import Enum
from typing import Optional
import os


class VoiceMode(Enum):
    """Available voice system modes."""

    REALTIME = "realtime"      # OpenAI Realtime API (~$0.30/min)
    HYBRID = "hybrid"          # Qwen + Claude hybrid (~$0.01/complex request)
    TRADITIONAL = "traditional"  # Whisper → Claude → TTS (medium cost)
    LOCAL_ONLY = "local"       # Fully local with Ollama (FREE)


def get_voice_system(mode: Optional[VoiceMode] = None):
    """
    Get the appropriate voice system based on mode.

    Args:
        mode: VoiceMode to use. If None, uses config settings.

    Returns:
        Voice system instance ready to use.
    """
    from .config import (
        REALTIME_VOICE_ENABLED,
        HYBRID_VOICE_ENABLED,
    )

    # Auto-detect mode from config if not specified
    if mode is None:
        if HYBRID_VOICE_ENABLED:
            mode = VoiceMode.HYBRID
        elif REALTIME_VOICE_ENABLED:
            mode = VoiceMode.REALTIME
        else:
            mode = VoiceMode.TRADITIONAL

    if mode == VoiceMode.REALTIME:
        from .realtime_voice import RealtimeConversationLoop
        from .config import OPENAI_API_KEY, REALTIME_VOICE_MODEL, REALTIME_VOICE_VOICE

        from .realtime_voice import RealtimeConfig
        config = RealtimeConfig(
            model=REALTIME_VOICE_MODEL,
            voice=REALTIME_VOICE_VOICE,
        )
        return RealtimeConversationLoop(api_key=OPENAI_API_KEY, config=config)

    elif mode == VoiceMode.HYBRID:
        from .qwen_voice import QwenClaudeHybrid, QwenVoiceConfig
        from .config import (
            HYBRID_QWEN_MODEL,
            HYBRID_QWEN_QUANTIZATION,
            HYBRID_USE_LOCAL_FOR_SIMPLE,
            CLAUDE_MODEL,
            CLAUDE_MODEL_COMPLEX,
        )

        config = QwenVoiceConfig(
            qwen_model=HYBRID_QWEN_MODEL,
            qwen_quantization=HYBRID_QWEN_QUANTIZATION,
            claude_model=CLAUDE_MODEL,
            claude_model_complex=CLAUDE_MODEL_COMPLEX,
            use_local_for_simple=HYBRID_USE_LOCAL_FOR_SIMPLE,
        )
        return QwenClaudeHybrid(config)

    elif mode == VoiceMode.LOCAL_ONLY:
        from .qwen_voice import QwenClaudeHybrid, QwenVoiceConfig

        # Configure for fully local operation
        config = QwenVoiceConfig(
            use_local_for_simple=True,
        )
        hybrid = QwenClaudeHybrid(config)

        # Override to never use Claude
        original_classify = hybrid.router.classify
        def always_local(text):
            return ("local", {"action": "unknown", "text": text})
        hybrid.router.classify = always_local

        return hybrid

    else:  # TRADITIONAL
        # Fall back to Gemini hybrid or basic pipeline
        try:
            from .hybrid_voice import HybridVoiceSystem
            return HybridVoiceSystem()
        except ImportError:
            raise NotImplementedError("Traditional voice mode not yet implemented")


def compare_costs():
    """Print a cost comparison of different voice modes."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              ARIA VOICE SYSTEM COST COMPARISON                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Mode            │ Cost/Min  │ Latency │ Quality │ Offline?     ║
║  ────────────────┼───────────┼─────────┼─────────┼────────────  ║
║  REALTIME        │ ~$0.30    │ 230ms   │ ★★★★★   │ No           ║
║  (OpenAI)        │           │         │         │              ║
║  ────────────────┼───────────┼─────────┼─────────┼────────────  ║
║  HYBRID          │ ~$0.01*   │ 500ms   │ ★★★★☆   │ Partial      ║
║  (Qwen+Claude)   │           │         │         │              ║
║  ────────────────┼───────────┼─────────┼─────────┼────────────  ║
║  LOCAL_ONLY      │ $0.00     │ 1-2s    │ ★★★☆☆   │ Yes          ║
║  (Qwen+Ollama)   │           │         │         │              ║
║                                                                  ║
║  * Complex queries only; simple commands are FREE                ║
║                                                                  ║
║  Monthly estimate (2hr/day active use):                          ║
║    REALTIME:   ~$1,080/month                                     ║
║    HYBRID:     ~$30-60/month                                     ║
║    LOCAL_ONLY: $0/month                                          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


async def demo():
    """Demo the hybrid voice system."""
    print("\nStarting Qwen + Claude Hybrid Voice System...\n")

    voice = get_voice_system(VoiceMode.HYBRID)

    # Set up callbacks
    voice.on_transcription = lambda t: print(f"[Heard]: {t}")
    voice.on_response = lambda r: print(f"[Aria]: {r}")
    voice.on_action = lambda a: print(f"[Action]: {a}")

    try:
        await voice.start_listening()
    except KeyboardInterrupt:
        voice.stop()


if __name__ == "__main__":
    import asyncio

    compare_costs()

    response = input("\nStart demo? [y/N]: ").strip().lower()
    if response == "y":
        asyncio.run(demo())
