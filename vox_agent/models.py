"""Core data types shared across the pipeline.

These are plain dataclasses (no third-party deps) so they serialise cleanly to
JSON for the WebSocket protocol and stay cheap to create in the hot path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class AudioFormat(str, Enum):
    """Wire formats we support. All mono."""

    PCM16 = "pcm16"      # 16-bit signed little-endian linear PCM
    MULAW = "mulaw"      # 8-bit G.711 mu-law (Twilio phone lines)


@dataclass
class AudioChunk:
    """A slice of mono audio flowing in either direction."""

    data: bytes
    sample_rate: int = 16000
    fmt: AudioFormat = AudioFormat.PCM16
    ts: float = field(default_factory=time.monotonic)

    @property
    def num_samples(self) -> int:
        return len(self.data) // (1 if self.fmt is AudioFormat.MULAW else 2)

    @property
    def duration_ms(self) -> float:
        if self.sample_rate == 0:
            return 0.0
        return 1000.0 * self.num_samples / self.sample_rate


@dataclass
class Transcript:
    """A transcription result. `is_final` distinguishes interim vs. settled text."""

    text: str
    is_final: bool = False
    confidence: float = 1.0
    ts: float = field(default_factory=time.monotonic)


@dataclass
class Token:
    """A single streamed LLM token (or small text delta)."""

    text: str
    ts: float = field(default_factory=time.monotonic)


@dataclass
class ToolCall:
    """A function-call request emitted by the LLM."""

    name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass
class ToolResult:
    """The value returned after running a ToolCall."""

    call_id: str
    name: str
    content: Any
    ok: bool = True
    error: Optional[str] = None


@dataclass
class Turn:
    """One entry in the conversation history."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_result: Optional[ToolResult] = None
    ts: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {"name": c.name, "arguments": c.arguments, "call_id": c.call_id}
                for c in self.tool_calls
            ]
        if self.tool_result is not None:
            d["tool_result"] = {
                "name": self.tool_result.name,
                "content": self.tool_result.content,
                "ok": self.tool_result.ok,
            }
        return d


@dataclass
class ConversationState:
    """Short-term memory for a single call/session.

    Held in Redis in production (see ``vox_agent.state``) but the default
    backend is an in-process dict so everything runs offline.
    """

    session_id: str
    history: list[Turn] = field(default_factory=list)
    slots: dict[str, Any] = field(default_factory=dict)
    stage: str = "greet"
    summary: str = ""
    created: float = field(default_factory=time.time)

    def add(self, turn: Turn) -> None:
        self.history.append(turn)

    def last_user_text(self) -> str:
        for t in reversed(self.history):
            if t.role is Role.USER:
                return t.content
        return ""
