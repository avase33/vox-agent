"""ElevenLabs streaming TTS adapter (optional, real API).

Only imported when ``VOX_TTS=elevenlabs``. Requires ``pip install vox-agent[elevenlabs]``
and ``ELEVENLABS_API_KEY``.
"""

from __future__ import annotations

from typing import AsyncIterator

from ..models import AudioChunk, AudioFormat
from .base import BaseTTS


class ElevenLabsTTS(BaseTTS):
    name = "elevenlabs"

    def __init__(self, api_key: str, voice: str = "Rachel", sample_rate: int = 16000) -> None:
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is required for the ElevenLabs TTS adapter")
        try:
            from elevenlabs.client import AsyncElevenLabs  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Run: pip install 'vox-agent[elevenlabs]'") from e
        self._client = AsyncElevenLabs(api_key=api_key)
        self._voice = voice
        self.sample_rate = sample_rate

    async def stream(self, text: str) -> AsyncIterator[AudioChunk]:  # pragma: no cover
        if not text.strip():
            return
        audio_stream = self._client.text_to_speech.stream(
            voice_id=self._voice,
            text=text,
            model_id="eleven_turbo_v2_5",
            output_format="pcm_16000",
        )
        async for data in audio_stream:
            if data:
                yield AudioChunk(data=data, sample_rate=self.sample_rate, fmt=AudioFormat.PCM16)
