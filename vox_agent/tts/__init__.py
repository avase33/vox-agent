"""TTS provider registry / factory."""

from __future__ import annotations

from ..config import Settings
from .base import BaseTTS
from .mock import MockTTS


def build_tts(settings: Settings) -> BaseTTS:
    provider = settings.tts_provider.lower()
    if provider in ("mock", "offline", ""):
        return MockTTS(sample_rate=settings.sample_rate, frame_ms=settings.frame_ms)
    if provider == "cartesia":
        from .cartesia import CartesiaTTS

        return CartesiaTTS(settings.cartesia_api_key, settings.tts_voice, settings.sample_rate)
    if provider == "elevenlabs":
        from .elevenlabs import ElevenLabsTTS

        return ElevenLabsTTS(settings.elevenlabs_api_key, settings.tts_voice, settings.sample_rate)
    raise ValueError(f"unknown TTS provider: {settings.tts_provider!r}")


__all__ = ["BaseTTS", "MockTTS", "build_tts"]
