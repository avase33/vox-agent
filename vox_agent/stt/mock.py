"""Offline mock STT — a deterministic streaming-recogniser simulator.

Because we can't run a real acoustic model with zero dependencies, the mock
takes the utterance's ground-truth text (the ``oracle``, provided by the
simulated caller) and *streams it back the way a real recogniser would*: interim
hypotheses that grow word-by-word, then a settled final. The pacing is derived
from the audio length so latency numbers stay realistic.

If no oracle is supplied (e.g. real mic audio with the mock selected), it falls
back to emitting a placeholder whose length scales with the speech energy, which
is enough to keep the pipeline flowing during smoke tests.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

from ..audio.pcm import rms_energy
from ..models import Transcript
from .base import BaseSTT


class MockSTT(BaseSTT):
    name = "mock"

    def __init__(self, interim_delay_s: float = 0.01) -> None:
        self.interim_delay_s = interim_delay_s

    async def transcribe(
        self, audio: bytes, sample_rate: int, oracle: Optional[str] = None
    ) -> AsyncIterator[Transcript]:
        if oracle is None:
            oracle = self._placeholder(audio)

        words = oracle.split()
        acc: list[str] = []
        for w in words:
            acc.append(w)
            await asyncio.sleep(self.interim_delay_s)
            yield Transcript(text=" ".join(acc), is_final=False, confidence=0.6)
        yield Transcript(text=oracle.strip(), is_final=True, confidence=0.97)

    @staticmethod
    def _placeholder(audio: bytes) -> str:
        # crude: number of "syllables" ~ speech duration; used only w/o oracle.
        n = max(1, len(audio) // 3200)  # ~200ms blocks at 16kHz PCM16
        energy = rms_energy(audio)
        if energy < 200:
            return ""
        return " ".join(["(speech)"] * min(n, 12))
