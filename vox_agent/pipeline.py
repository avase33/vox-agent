"""The orchestrator — wires hearing, thinking and speaking into one turn loop.

Data flow per turn::

    audio --> STT --> PII redact --> LLM (streaming, function calling)
                                       |-> ToolRegistry (act on the world)
                                       '-> tokens --> clause buffer --> TTS --> audio out

Everything is an async generator so callers (CLI, WebSocket server, tests) can
consume :class:`AgentEvent` s as they happen. Two properties make it feel human:

* **Streaming speech** — the first clause is spoken before the LLM finishes the
  sentence, so time-to-first-byte stays low.
* **Barge-in** — :meth:`VoiceAgent.interrupt` sets a per-session flag that
  instantly halts token generation and flushes the TTS buffer, so the agent stops
  talking the moment the caller does.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from .config import Settings
from .dialog.context import build_prompt_turns
from .dialog.pii import redact
from .llm import build_llm
from .metrics import MetricsCollector, TurnMetrics
from .models import ConversationState, Role, ToolCall, Turn
from .state import build_store
from .stt import build_stt
from .tools import ToolRegistry
from .tts import build_tts

_BOUNDARY = tuple(".!?,;:")
_MAX_TOOL_HOPS = 6


def _clause_boundary(text: str) -> bool:
    t = text.rstrip()
    if not t:
        return False
    if t[-1] in _BOUNDARY:
        return True
    return len(text.split()) >= 12


@dataclass
class AgentEvent:
    """A single thing that happened during a turn, streamed to the caller."""

    type: str  # transcript|user_final|redacted|token|tool|tool_result|audio|turn_done|interrupted|call_end
    text: str = ""
    audio: bytes = b""
    data: dict[str, Any] = field(default_factory=dict)
    metrics: Optional[dict[str, Any]] = None


class VoiceAgent:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.stt = build_stt(self.settings)
        self.llm = build_llm(self.settings)
        self.tts = build_tts(self.settings)
        self.tools = ToolRegistry()
        self.store = build_store(self.settings)
        self.metrics = MetricsCollector()
        self._interrupts: dict[str, asyncio.Event] = {}

    # -- interruption -----------------------------------------------------
    def _flag(self, session_id: str) -> asyncio.Event:
        return self._interrupts.setdefault(session_id, asyncio.Event())

    def interrupt(self, session_id: str) -> None:
        """Barge-in: stop the agent talking right now."""
        self._flag(session_id).set()

    # -- public turn entry points ----------------------------------------
    async def open_call(self, session_id: str) -> AsyncIterator[AgentEvent]:
        """Agent speaks first (the greeting)."""
        state = self.store.get(session_id)
        self._flag(session_id).clear()
        async for ev in self._respond(state, None):
            yield ev

    async def handle_utterance(
        self, session_id: str, audio: bytes, oracle: Optional[str] = None
    ) -> AsyncIterator[AgentEvent]:
        """Process one user utterance end-to-end and stream the reply."""
        state = self.store.get(session_id)
        self._flag(session_id).clear()
        m = self.metrics.start_turn(session_id)

        final_text = ""
        async for tr in self.stt.transcribe(audio, self.settings.sample_rate, oracle):
            if tr.is_final:
                final_text = tr.text
                m.t_stt_final = time.monotonic()
                yield AgentEvent("user_final", text=final_text)
            else:
                yield AgentEvent("transcript", text=tr.text)

        if self.settings.redact_pii and final_text:
            red = redact(final_text)
            final_text = red.text
            if red.vault:
                state.slots.setdefault("_vault", {}).update(red.vault)
            if red.redactions:
                yield AgentEvent("redacted", data={"types": red.redactions})

        state.add(Turn(Role.USER, final_text))
        async for ev in self._respond(state, m):
            yield ev

    async def handle_text(
        self, session_id: str, text: str
    ) -> AsyncIterator[AgentEvent]:
        """Process an already-transcribed user turn (skips STT).

        Used by clients that transcribe on-device (e.g. the browser's Web Speech
        API) or by the text CLI. PII redaction still applies.
        """
        state = self.store.get(session_id)
        self._flag(session_id).clear()
        m = self.metrics.start_turn(session_id)
        m.t_stt_final = time.monotonic()
        yield AgentEvent("user_final", text=text)

        if self.settings.redact_pii and text:
            red = redact(text)
            text = red.text
            if red.vault:
                state.slots.setdefault("_vault", {}).update(red.vault)
            if red.redactions:
                yield AgentEvent("redacted", data={"types": red.redactions})

        state.add(Turn(Role.USER, text))
        async for ev in self._respond(state, m):
            yield ev

    # -- core response loop ----------------------------------------------
    async def _respond(
        self, state: ConversationState, m: Optional[TurnMetrics]
    ) -> AsyncIterator[AgentEvent]:
        flag = self._flag(state.session_id)
        reply_parts: list[str] = []
        clause: list[str] = []

        for _ in range(_MAX_TOOL_HOPS):
            tool_ran = False
            async for ev in self.llm.stream(state, self.tools.schemas):
                if flag.is_set():
                    break

                if isinstance(ev, ToolCall):
                    tool_ran = True
                    if m:
                        m.tool_calls += 1
                    state.add(Turn(Role.ASSISTANT, "".join(clause), tool_calls=[ev]))
                    clause = []
                    yield AgentEvent("tool", data={"name": ev.name, "arguments": ev.arguments})
                    result = self.tools.run(ev)
                    state.add(Turn(Role.TOOL, tool_result=result))
                    yield AgentEvent(
                        "tool_result",
                        data={"name": result.name, "content": result.content, "ok": result.ok},
                    )
                    break

                # Token
                if m and not m.t_first_token:
                    m.t_first_token = time.monotonic()
                reply_parts.append(ev.text)
                clause.append(ev.text)
                yield AgentEvent("token", text=ev.text)

                buffered = "".join(clause)
                if _clause_boundary(buffered):
                    async for aev in self._speak(buffered, m, flag):
                        yield aev
                    clause = []

            if flag.is_set() or not tool_ran:
                break

        if clause and not flag.is_set():
            async for aev in self._speak("".join(clause), m, flag):
                yield aev

        full_reply = "".join(reply_parts).strip()
        if full_reply:
            state.add(Turn(Role.ASSISTANT, full_reply))
        # keep the model context bounded on long calls
        build_prompt_turns(state, self.settings.context_window_turns)
        self.store.save(state)

        if flag.is_set():
            if m:
                m.interrupted = True
                m.t_end = time.monotonic()
            yield AgentEvent("interrupted", metrics=(m.to_dict() if m else None))
            return

        if m:
            m.t_end = time.monotonic()
        yield AgentEvent("turn_done", text=full_reply, metrics=(m.to_dict() if m else None))
        if state.slots.get("end_call"):
            yield AgentEvent("call_end")

    async def _speak(
        self, text: str, m: Optional[TurnMetrics], flag: asyncio.Event
    ) -> AsyncIterator[AgentEvent]:
        async for chunk in self.tts.stream(text):
            if flag.is_set():
                break
            if m and not m.t_first_audio:
                m.t_first_audio = time.monotonic()
            yield AgentEvent("audio", audio=chunk.data, data={"sample_rate": chunk.sample_rate})
