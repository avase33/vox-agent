"""Offline call simulation harness.

Turns a script of caller lines into synthetic speech audio (real PCM with real
energy, so the VAD and audio path are genuinely exercised) and drives a full call
through :class:`VoiceAgent`, returning every :class:`AgentEvent`. This is how the
whole system runs — hearing, thinking, acting, speaking — with zero API keys and
no microphone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .audio.synth import synth_utterance
from .config import Settings
from .pipeline import AgentEvent, VoiceAgent

# A representative appointment-booking call.
BOOKING_SCRIPT = [
    "Hi, I'd like to book an appointment.",
    "Wednesday works for me.",
    "Let's do 11am.",
    "My name is Jordan Lee.",
    "Yes, please book it.",
    "No, that's all. Thank you!",
]

# A caller who gives everything at once, then interrupts the confirmation.
FAST_SCRIPT = [
    "Can I book Wednesday at 3pm, my name is Sam Rivera.",
    "Yes that's perfect.",
]

SCRIPTS = {"booking": BOOKING_SCRIPT, "fast": FAST_SCRIPT}


@dataclass
class CallResult:
    events: list[AgentEvent] = field(default_factory=list)
    transcript_lines: list[str] = field(default_factory=list)

    def spoken_text(self) -> str:
        return " ".join(
            e.text for e in self.events if e.type == "turn_done" and e.text
        ).strip()

    def audio_bytes(self) -> int:
        return sum(len(e.audio) for e in self.events if e.type == "audio")

    def tool_calls(self) -> list[dict]:
        return [e.data for e in self.events if e.type == "tool"]


async def run_call(
    agent: VoiceAgent,
    session_id: str,
    script: list[str],
    settings: Optional[Settings] = None,
) -> CallResult:
    settings = settings or agent.settings
    result = CallResult()

    async for ev in agent.open_call(session_id):
        result.events.append(ev)
        if ev.type == "turn_done" and ev.text:
            result.transcript_lines.append(f"agent: {ev.text}")

    for line in script:
        audio = synth_utterance(line, settings.sample_rate)
        result.transcript_lines.append(f"caller: {line}")
        async for ev in agent.handle_utterance(session_id, audio, oracle=line):
            result.events.append(ev)
            if ev.type == "turn_done" and ev.text:
                result.transcript_lines.append(f"agent: {ev.text}")
            if ev.type == "call_end":
                return result
    return result


async def run_call_with_interrupt(
    agent: VoiceAgent, session_id: str, line: str, interrupt_after_tokens: int = 3
) -> CallResult:
    """Speak one line; barge in after a few reply tokens to prove interruption."""
    result = CallResult()
    async for ev in agent.open_call(session_id):
        result.events.append(ev)

    audio = synth_utterance(line, agent.settings.sample_rate)
    token_count = 0
    async for ev in agent.handle_utterance(session_id, audio, oracle=line):
        result.events.append(ev)
        if ev.type == "token":
            token_count += 1
            if token_count == interrupt_after_tokens:
                agent.interrupt(session_id)
    return result
