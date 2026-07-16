"""Deepgram streaming STT adapter (optional, real API).

Only imported when ``VOX_STT=deepgram``. Requires ``pip install vox-agent[deepgram]``
and ``DEEPGRAM_API_KEY``. Kept intentionally small: it batches the utterance to
Deepgram's pre-recorded endpoint via the SDK. For a true token-by-token feed you
would use Deepgram's live WebSocket; the streaming interface here is preserved so
that swap is local to this file.
"""

from __future__ import annotations

from typing import AsyncIterator, Optional

from ..models import Transcript
from .base import BaseSTT


class DeepgramSTT(BaseSTT):
    name = "deepgram"

    def __init__(self, api_key: str, model: str = "nova-2") -> None:
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is required for the Deepgram STT adapter")
        try:
            from deepgram import DeepgramClient  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "deepgram-sdk not installed. Run: pip install 'vox-agent[deepgram]'"
            ) from e
        self._client = DeepgramClient(api_key)
        self._model = model

    async def transcribe(
        self, audio: bytes, sample_rate: int, oracle: Optional[str] = None
    ) -> AsyncIterator[Transcript]:  # pragma: no cover - needs network + key
        from deepgram import PrerecordedOptions  # type: ignore

        source = {"buffer": audio, "mimetype": "audio/l16;rate=%d" % sample_rate}
        options = PrerecordedOptions(model=self._model, smart_format=True, language="en")
        resp = await self._client.listen.asyncrest.v("1").transcribe_file(source, options)
        text = resp["results"]["channels"][0]["alternatives"][0]["transcript"]
        yield Transcript(text=text, is_final=True, confidence=0.95)
