"""Energy-based Voice Activity Detection with hangover / endpointing.

A production system would use a neural VAD (e.g. Silero), but an energy VAD with
a start-count and an end-silence hangover is enough to demonstrate correct
endpointing and, crucially, *barge-in* detection: knowing the exact instant the
user starts speaking while the agent is talking.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .pcm import rms_energy


class VADState(str, Enum):
    SILENCE = "silence"
    SPEECH = "speech"


@dataclass
class VADEvent:
    kind: str          # "speech_start" | "speech_end" | None
    state: VADState
    energy: float


class VoiceActivityDetector:
    """Feed fixed-size PCM16 frames; get speech-start / speech-end events.

    - ``start_frames`` consecutive loud frames flip SILENCE -> SPEECH.
    - ``end_silence_ms`` of quiet flips SPEECH -> SILENCE (endpoint).
    """

    def __init__(
        self,
        threshold: float = 500.0,
        start_frames: int = 3,
        end_silence_ms: int = 480,
        frame_ms: int = 20,
    ) -> None:
        self.threshold = threshold
        self.start_frames = max(1, start_frames)
        self.end_silence_frames = max(1, int(end_silence_ms / max(1, frame_ms)))
        self.state = VADState.SILENCE
        self._loud_run = 0
        self._quiet_run = 0
        self.last_energy = 0.0

    def reset(self) -> None:
        self.state = VADState.SILENCE
        self._loud_run = 0
        self._quiet_run = 0

    def is_speaking(self) -> bool:
        return self.state is VADState.SPEECH

    def process(self, frame: bytes) -> VADEvent:
        energy = rms_energy(frame)
        self.last_energy = energy
        loud = energy >= self.threshold

        if self.state is VADState.SILENCE:
            if loud:
                self._loud_run += 1
                if self._loud_run >= self.start_frames:
                    self.state = VADState.SPEECH
                    self._quiet_run = 0
                    return VADEvent("speech_start", self.state, energy)
            else:
                self._loud_run = 0
        else:  # SPEECH
            if loud:
                self._quiet_run = 0
            else:
                self._quiet_run += 1
                if self._quiet_run >= self.end_silence_frames:
                    self.state = VADState.SILENCE
                    self._loud_run = 0
                    return VADEvent("speech_end", self.state, energy)

        return VADEvent("", self.state, energy)
