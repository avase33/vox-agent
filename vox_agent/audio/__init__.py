"""From-scratch audio utilities: PCM/mu-law codecs, resampling, RMS energy,
tone synthesis, an energy VAD and a client-side jitter buffer. No numpy."""

from .pcm import (
    downsample,
    mulaw_decode,
    mulaw_encode,
    pcm_bytes_to_ints,
    ints_to_pcm_bytes,
    rms_energy,
)
from .synth import beep, synth_utterance, wav_bytes
from .vad import VADState, VoiceActivityDetector
from .jitter import JitterBuffer

__all__ = [
    "downsample",
    "mulaw_decode",
    "mulaw_encode",
    "pcm_bytes_to_ints",
    "ints_to_pcm_bytes",
    "rms_energy",
    "beep",
    "synth_utterance",
    "wav_bytes",
    "VADState",
    "VoiceActivityDetector",
    "JitterBuffer",
]
