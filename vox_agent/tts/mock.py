"""Offline mock TTS.

Produces *real* PCM16 audio (see ``vox_agent.audio.synth``) so the outbound audio
path — chunking, WebSocket framing, jitter buffering, playback — is fully
exercised with no paid API. It streams frame-by-frame with a small per-frame delay
to imitate a real streaming synthesiser, which is also what lets barge-in
interruption cut it off mid-word.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from ..audio.synth import synth_utterance
from ..models import AudioChunk, AudioFormat
from .base import BaseTTS


class MockTTS(BaseTTS):
    name = "mock"

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 20, realtime: bool = True) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.realtime = realtime
        self._frame_bytes = int(sample_rate * frame_ms / 1000) * 2

    async def stream(self, text: str) -> AsyncIterator[AudioChunk]:
        if not text.strip():
            return
        pcm = synth_utterance(text, self.sample_rate)
        for i in range(0, len(pcm), self._frame_bytes):
            frame = pcm[i : i + self._frame_bytes]
            if self.realtime:
                # emit slightly faster than real-time so we never underrun
                await asyncio.sleep(self.frame_ms / 1000 * 0.5)
            yield AudioChunk(data=frame, sample_rate=self.sample_rate, fmt=AudioFormat.PCM16)
