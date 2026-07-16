"""STT provider registry / factory."""

from __future__ import annotations

from ..config import Settings
from .base import BaseSTT
from .mock import MockSTT


def build_stt(settings: Settings) -> BaseSTT:
    provider = settings.stt_provider.lower()
    if provider in ("mock", "offline", ""):
        return MockSTT()
    if provider == "deepgram":
        from .deepgram import DeepgramSTT

        return DeepgramSTT(settings.deepgram_api_key)
    raise ValueError(f"unknown STT provider: {settings.stt_provider!r}")


__all__ = ["BaseSTT", "MockSTT", "build_stt"]
