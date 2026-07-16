"""PCM / mu-law codecs, resampling and energy — implemented from scratch.

We deliberately avoid numpy so the core has zero dependencies and installs in
one second. These functions are only used on small frames (10-20 ms), so pure
Python is more than fast enough for real-time.
"""

from __future__ import annotations

import array
import math

_BIAS = 0x84
_CLIP = 32635


def pcm_bytes_to_ints(data: bytes) -> array.array:
    """Decode signed 16-bit little-endian PCM into an int array."""
    samples = array.array("h")
    samples.frombytes(data[: len(data) - (len(data) % 2)])
    # `h` is native-endian; normalise to little-endian semantics.
    import sys

    if sys.byteorder == "big":
        samples.byteswap()
    return samples


def ints_to_pcm_bytes(samples: "array.array | list[int]") -> bytes:
    """Encode ints back to signed 16-bit little-endian PCM bytes."""
    arr = array.array("h", (max(-32768, min(32767, int(s))) for s in samples))
    import sys

    if sys.byteorder == "big":
        arr.byteswap()
    return arr.tobytes()


def rms_energy(data: bytes) -> float:
    """Root-mean-square amplitude of a PCM16 frame (0..32767)."""
    samples = pcm_bytes_to_ints(data)
    if not samples:
        return 0.0
    acc = 0
    for s in samples:
        acc += s * s
    return math.sqrt(acc / len(samples))


def downsample(data: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Naive integer-factor decimation of PCM16 (good enough for VAD/STT feed)."""
    if dst_rate <= 0 or src_rate <= 0 or src_rate == dst_rate:
        return data
    samples = pcm_bytes_to_ints(data)
    ratio = src_rate / dst_rate
    out: list[int] = []
    idx = 0.0
    while int(idx) < len(samples):
        out.append(samples[int(idx)])
        idx += ratio
    return ints_to_pcm_bytes(out)


def mulaw_encode(data: bytes) -> bytes:
    """G.711 mu-law encode a PCM16 buffer -> 8-bit bytes (Twilio phone format)."""
    samples = pcm_bytes_to_ints(data)
    out = bytearray()
    for sample in samples:
        sign = 0x80 if sample < 0 else 0x00
        if sample < 0:
            sample = -sample
        if sample > _CLIP:
            sample = _CLIP
        sample += _BIAS
        exponent = 7
        mask = 0x4000
        while exponent > 0 and not (sample & mask):
            exponent -= 1
            mask >>= 1
        mantissa = (sample >> (exponent + 3)) & 0x0F
        out.append(~(sign | (exponent << 4) | mantissa) & 0xFF)
    return bytes(out)


def mulaw_decode(data: bytes) -> bytes:
    """G.711 mu-law decode 8-bit bytes -> PCM16."""
    out: list[int] = []
    for byte in data:
        byte = ~byte & 0xFF
        sign = byte & 0x80
        exponent = (byte >> 4) & 0x07
        mantissa = byte & 0x0F
        sample = ((mantissa << 3) + _BIAS) << exponent
        sample -= _BIAS
        out.append(-sample if sign else sample)
    return ints_to_pcm_bytes(out)
