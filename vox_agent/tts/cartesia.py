"""Cartesia Sonic streaming TTS adapter (optional, real API).

Only imported when ``VOX_TTS=cartesia``. Requires ``pip install vox-agent[cartesia]``
and ``CARTESIA_API_KEY``. Streams raw PCM frames back as they arrive.
"""

from __future__ import annotations

from typing import AsyncIterator

from ..models import AudioChunk, AudioFormat
from .base import BaseTTS


class CartesiaTTS(BaseTTS):
    name = "cartesia"

    def __init__(self, api_key: str, voice: str = "default", sample_rate: int = 16000) -> None:
        if not api_key:
            raise RuntimeError("CARTESIA_API_KEY is required for the Cartesia TTS adapter")
        try:
            from cartesia import AsyncCartesia  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Run: pip install 'vox-agent[cartesia]'") from e
        self._client = AsyncCartesia(api_key=api_key)
        self._voice = voice
        self.sample_rate = sample_rate

    async def stream(self, text: str) -> AsyncIterator[AudioChunk]:  # pragma: no cover
        if not text.strip():
            return
        gen = self._client.tts.sse(
            model_id="sonic-english",
            transcript=text,
            voice={"mode": "id", "id": self._voice} if self._voice != "default" else None,
            output_format={
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.sample_rate,
            },
        )
        async for event in gen:
            data = getattr(event, "audio", None) or (event.get("audio") if isinstance(event, dict) else None)
            if data:
                yield AudioChunk(data=data, sample_rate=self.sample_rate, fmt=AudioFormat.PCM16)
