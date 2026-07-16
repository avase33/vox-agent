"""Text-to-speech provider interface.

A TTS provider turns a short text chunk (a clause the pipeline has buffered from
the LLM stream) into a stream of :class:`AudioChunk` frames, so playback can
begin before the whole sentence exists. ``time_to_first_byte`` is the metric that
matters most here.
"""

from __future__ import annotations

import abc
from typing import AsyncIterator

from ..models import AudioChunk


class BaseTTS(abc.ABC):
    name: str = "base"
    sample_rate: int = 16000

    @abc.abstractmethod
    async def stream(self, text: str) -> AsyncIterator[AudioChunk]:
        raise NotImplementedError
        yield  # pragma: no cover

    async def aclose(self) -> None:  # pragma: no cover
        return None
