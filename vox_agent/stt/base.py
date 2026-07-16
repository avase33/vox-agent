"""Speech-to-text provider interface.

An STT provider transcribes the PCM audio of a *single utterance* and streams
back :class:`~vox_agent.models.Transcript` results — zero or more interim
(``is_final=False``) results followed by exactly one final result. Endpointing
(deciding when an utterance ends) is done upstream by the VAD, so providers only
worry about turning audio into words.
"""

from __future__ import annotations

import abc
from typing import AsyncIterator, Optional

from ..models import Transcript


class BaseSTT(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    async def transcribe(
        self, audio: bytes, sample_rate: int, oracle: Optional[str] = None
    ) -> AsyncIterator[Transcript]:
        """Yield interim transcripts then one final transcript.

        ``oracle`` is an optional ground-truth transcript used *only* by the
        offline mock provider to simulate a perfect recogniser; real providers
        ignore it and transcribe the audio for real.
        """
        raise NotImplementedError
        yield  # pragma: no cover  (makes this an async generator)

    async def aclose(self) -> None:  # pragma: no cover - override if needed
        return None
