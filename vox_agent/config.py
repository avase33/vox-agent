"""Runtime configuration.

Everything is driven by environment variables so the same image runs offline
(all ``mock`` providers, zero keys) or against real APIs by flipping a few vars.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    # --- provider selection: "mock" (offline) | real adapter name ---------
    stt_provider: str = field(default_factory=lambda: _env("VOX_STT", "mock"))
    llm_provider: str = field(default_factory=lambda: _env("VOX_LLM", "mock"))
    tts_provider: str = field(default_factory=lambda: _env("VOX_TTS", "mock"))
    state_backend: str = field(default_factory=lambda: _env("VOX_STATE", "memory"))

    # --- audio -----------------------------------------------------------
    sample_rate: int = field(default_factory=lambda: _env_int("VOX_SAMPLE_RATE", 16000))
    frame_ms: int = field(default_factory=lambda: _env_int("VOX_FRAME_MS", 20))

    # --- VAD / endpointing ----------------------------------------------
    vad_energy_threshold: float = field(
        default_factory=lambda: _env_float("VOX_VAD_THRESHOLD", 500.0)
    )
    vad_start_frames: int = field(default_factory=lambda: _env_int("VOX_VAD_START_FRAMES", 3))
    vad_end_silence_ms: int = field(default_factory=lambda: _env_int("VOX_VAD_END_SILENCE_MS", 480))

    # --- dialog ----------------------------------------------------------
    scenario: str = field(default_factory=lambda: _env("VOX_SCENARIO", "receptionist"))
    context_window_turns: int = field(default_factory=lambda: _env_int("VOX_CTX_TURNS", 10))
    redact_pii: bool = field(default_factory=lambda: _env("VOX_REDACT_PII", "1") == "1")

    # --- keys (only read by real adapters) -------------------------------
    deepgram_api_key: str = field(default_factory=lambda: _env("DEEPGRAM_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    cartesia_api_key: str = field(default_factory=lambda: _env("CARTESIA_API_KEY", ""))
    elevenlabs_api_key: str = field(default_factory=lambda: _env("ELEVENLABS_API_KEY", ""))
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0"))

    # --- model names -----------------------------------------------------
    llm_model: str = field(default_factory=lambda: _env("VOX_LLM_MODEL", "llama-3.1-8b-instant"))
    tts_voice: str = field(default_factory=lambda: _env("VOX_TTS_VOICE", "default"))

    @property
    def frame_bytes(self) -> int:
        """Bytes per PCM16 frame at the configured frame size."""
        return int(self.sample_rate * self.frame_ms / 1000) * 2
