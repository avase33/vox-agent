"""Deterministic audio synthesis.

Used two ways:
  * the offline **mock TTS** turns text into real, playable PCM/WAV bytes so the
    end-to-end audio path is exercised without any paid API;
  * tests/verifiers synthesise fake "user speech" (word tone-bursts separated by
    silence) that the energy VAD can segment, so the whole pipeline runs
    head-to-tail with no microphone.

The synthesis is a simple formant-ish sum of sines with an amplitude envelope —
it will not sound like a human, but it is real audio with real energy.
"""

from __future__ import annotations

import io
import math
import struct
import wave

_TWO_PI = 2.0 * math.pi


def _tone(freq: float, ms: int, sample_rate: int, amp: float = 0.35) -> list[int]:
    n = int(sample_rate * ms / 1000)
    out: list[int] = []
    for i in range(n):
        t = i / sample_rate
        # two partials for a slightly voice-like timbre
        v = math.sin(_TWO_PI * freq * t) + 0.5 * math.sin(_TWO_PI * 2 * freq * t)
        # short attack/release envelope to avoid clicks
        env = min(1.0, i / (0.01 * sample_rate + 1), (n - i) / (0.01 * sample_rate + 1))
        out.append(int(max(-1.0, min(1.0, v * amp * env)) * 32767))
    return out


def _silence(ms: int, sample_rate: int) -> list[int]:
    return [0] * int(sample_rate * ms / 1000)


def _ints_to_bytes(samples: list[int]) -> bytes:
    return struct.pack("<%dh" % len(samples), *samples)


def beep(freq: float = 440.0, ms: int = 120, sample_rate: int = 16000) -> bytes:
    """A single PCM16 tone burst."""
    return _ints_to_bytes(_tone(freq, ms, sample_rate))


def synth_utterance(text: str, sample_rate: int = 16000, wpm: int = 170) -> bytes:
    """Render `text` as a burst-per-word PCM16 signal with inter-word silence.

    Pitch is derived from each word so the signal is deterministic and varied.
    Word/gap durations approximate the requested words-per-minute so latency
    numbers in the demo are realistic.
    """
    words = text.split()
    if not words:
        return b""
    ms_per_word = max(120, int(60000 / max(1, wpm)))
    samples: list[int] = _silence(80, sample_rate)
    for w in words:
        base = 110 + (sum(ord(c) for c in w) % 12) * 15  # 110..275 Hz-ish
        dur = min(420, ms_per_word + 8 * len(w))
        samples += _tone(base, dur, sample_rate)
        samples += _silence(70, sample_rate)
    samples += _silence(120, sample_rate)
    return _ints_to_bytes(samples)


def wav_bytes(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw PCM16 in a WAV container (handy for saving demo audio)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()
