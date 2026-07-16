"""Latency instrumentation.

Voice UX lives or dies on latency, so we measure the pieces that matter:
  * **STT** — audio end to final transcript.
  * **TTFT** — final transcript to the LLM's first token ("thinking" time).
  * **TTFB** — the number users feel: user-stopped-talking to first audio byte.
  * **total** — the full turn.

``MetricsCollector.summary`` reports p50/p95 so you can quote real numbers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def _ms(a: float, b: float) -> float:
    return round((b - a) * 1000.0, 1) if (a and b and b >= a) else 0.0


@dataclass
class TurnMetrics:
    session_id: str
    t_user_end: float = 0.0     # audio finished / endpoint
    t_stt_final: float = 0.0    # final transcript ready
    t_first_token: float = 0.0  # LLM's first token
    t_first_audio: float = 0.0  # first TTS byte to client
    t_end: float = 0.0          # turn fully done
    tool_calls: int = 0
    interrupted: bool = False

    @property
    def stt_ms(self) -> float:
        return _ms(self.t_user_end, self.t_stt_final)

    @property
    def ttft_ms(self) -> float:
        return _ms(self.t_stt_final, self.t_first_token)

    @property
    def ttfb_ms(self) -> float:
        """The headline number: user stopped talking -> first audio out."""
        return _ms(self.t_user_end, self.t_first_audio)

    @property
    def total_ms(self) -> float:
        return _ms(self.t_user_end, self.t_end)

    def to_dict(self) -> dict[str, float | int | bool | str]:
        return {
            "session_id": self.session_id,
            "stt_ms": self.stt_ms,
            "ttft_ms": self.ttft_ms,
            "ttfb_ms": self.ttfb_ms,
            "total_ms": self.total_ms,
            "tool_calls": self.tool_calls,
            "interrupted": self.interrupted,
        }


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return round(s[k], 1)


@dataclass
class MetricsCollector:
    turns: list[TurnMetrics] = field(default_factory=list)

    def start_turn(self, session_id: str) -> TurnMetrics:
        m = TurnMetrics(session_id=session_id, t_user_end=time.monotonic())
        self.turns.append(m)
        return m

    def summary(self) -> dict[str, float | int]:
        done = [t for t in self.turns if t.t_end]
        ttfb = [t.ttfb_ms for t in done]
        total = [t.total_ms for t in done]
        return {
            "turns": len(done),
            "ttfb_p50_ms": _pct(ttfb, 50),
            "ttfb_p95_ms": _pct(ttfb, 95),
            "total_p50_ms": _pct(total, 50),
            "total_p95_ms": _pct(total, 95),
            "interruptions": sum(1 for t in done if t.interrupted),
            "tool_calls": sum(t.tool_calls for t in done),
        }
