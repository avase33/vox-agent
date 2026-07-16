#!/usr/bin/env python3
"""Full offline verification of vox-agent.

Exercises every subsystem — audio/VAD, PII redaction, tools + SQLite, the
streaming pipeline, function calling, barge-in interruption, context pruning and
latency — with zero API keys. Prints a pass/fail report.
"""

from __future__ import annotations

import asyncio
import sys

from vox_agent.audio import VoiceActivityDetector, mulaw_decode, mulaw_encode, synth_utterance
from vox_agent.dialog.context import build_prompt_turns
from vox_agent.dialog.pii import redact
from vox_agent.models import ConversationState, Role, Turn
from vox_agent.pipeline import VoiceAgent
from vox_agent.sim import SCRIPTS, run_call, run_call_with_interrupt
from vox_agent.tools import ToolRegistry
from vox_agent.models import ToolCall

_passed = 0
_failed = 0


def check(label: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  [PASS] {label}")
    else:
        _failed += 1
        print(f"  [FAIL] {label}")


def h(title: str) -> None:
    print(f"\n[{title}]")


async def main() -> int:
    print("=" * 72)
    print("vox-agent - full offline verification")
    print("=" * 72)
    print(f"python {sys.version.split()[0]}")

    h("1] audio + VAD")
    sr = 16000
    fb = int(sr * 0.02) * 2
    pcm = synth_utterance("please book an appointment", sr) + b"\x00" * (fb * 30)
    vad = VoiceActivityDetector(threshold=500, frame_ms=20)
    starts = ends = 0
    for i in range(0, len(pcm) - fb, fb):
        ev = vad.process(pcm[i : i + fb])
        starts += ev.kind == "speech_start"
        ends += ev.kind == "speech_end"
    check("VAD finds speech onset", starts >= 1)
    check("VAD endpoints on silence", ends >= 1)
    enc = mulaw_encode(pcm)
    check("mu-law halves byte count", len(enc) == len(pcm) // 2)
    check("mu-law decodes back to PCM16", len(mulaw_decode(enc)) == len(enc) * 2)

    h("2] PII redaction")
    r = redact("card 4111 1111 1111 1111 phone 415-555-0198 ssn 123-45-6789")
    check("card redacted", "[CARD_1]" in r.text and "4111 1111" not in r.text)
    check("phone redacted", "[PHONE_1]" in r.text)
    check("ssn redacted", "[SSN_1]" in r.text)
    check("last-4 retained for confirmation", any(v == "1111" for k, v in r.vault.items() if k.endswith(":last4")))

    h("3] tools + SQLite")
    reg = ToolRegistry()
    avail = reg.run(ToolCall("check_availability", {"day": "tuesday"}, "1"))
    check("availability excludes taken slots", "10:00" not in avail.content["open_slots"])
    booking = reg.run(ToolCall("create_booking", {"day": "wednesday", "time": "11am", "name": "Jo"}, "2"))
    check("booking persists", booking.content["ok"] and booking.content["time"] == "11:00")
    dbl = reg.run(ToolCall("create_booking", {"day": "wednesday", "time": "11:00", "name": "Al"}, "3"))
    check("double-booking rejected", dbl.content["ok"] is False)

    h("4] end-to-end booking call (STT -> LLM -> tools -> TTS)")
    agent = VoiceAgent()
    res = await run_call(agent, "v", SCRIPTS["booking"])
    tools = [t["name"] for t in res.tool_calls()]
    check("checked availability via function call", "check_availability" in tools)
    check("created booking via function call", "create_booking" in tools)
    check("streamed real audio out", res.audio_bytes() > 0)
    check("confirmed the booking to the caller", "all set" in " ".join(res.transcript_lines).lower())

    h("5] barge-in interruption")
    ires = await run_call_with_interrupt(agent, "int", "Can I book Wednesday at 11am?", 2)
    check("agent stops when interrupted", any(e.type == "interrupted" for e in ires.events))

    h("6] context pruning")
    st = ConversationState("c")
    for i in range(40):
        st.add(Turn(Role.USER, f"turn {i}"))
    pruned = build_prompt_turns(st, keep_recent=10)
    check("history windowed", len(pruned) <= 12)
    check("old turns summarised", bool(st.summary))

    h("7] latency")
    summary = agent.metrics.summary()
    print(f"    ttfb p50={summary['ttfb_p50_ms']}ms p95={summary['ttfb_p95_ms']}ms "
          f"total p95={summary['total_p95_ms']}ms over {summary['turns']} turns")
    check("time-to-first-byte under 500ms target", summary["ttfb_p95_ms"] < 500)

    print("\n" + "=" * 72)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 72)
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
