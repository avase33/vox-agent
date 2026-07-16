import pytest

from vox_agent.config import Settings
from vox_agent.pipeline import VoiceAgent
from vox_agent.sim import SCRIPTS, run_call, run_call_with_interrupt

pytestmark = pytest.mark.asyncio


async def test_booking_call_end_to_end():
    agent = VoiceAgent(Settings())
    res = await run_call(agent, "t", SCRIPTS["booking"])

    tools = [t["name"] for t in res.tool_calls()]
    assert "check_availability" in tools
    assert "create_booking" in tools

    # real audio was produced on the way out
    assert res.audio_bytes() > 0

    # the agent confirmed the booking
    transcript = " ".join(res.transcript_lines).lower()
    assert "all set" in transcript


async def test_agent_greets_first():
    agent = VoiceAgent(Settings())
    events = []
    async for ev in agent.open_call("g"):
        events.append(ev)
    said = " ".join(e.text for e in events if e.type == "turn_done")
    assert "book an appointment" in said.lower()


async def test_ttfb_is_low_offline():
    agent = VoiceAgent(Settings())
    await run_call(agent, "t2", SCRIPTS["booking"])
    summary = agent.metrics.summary()
    assert summary["turns"] >= 1
    # offline everything is in-process; comfortably under the 500ms target
    assert summary["ttfb_p95_ms"] < 500


async def test_pii_redacted_before_llm():
    agent = VoiceAgent(Settings())
    async for _ in agent.open_call("p"):
        pass
    redacted = []
    async for ev in agent.handle_text("p", "my card is 4111 1111 1111 1111"):
        if ev.type == "redacted":
            redacted.append(ev.data)
    assert redacted and "CARD" in redacted[0]["types"]
    # the raw number never entered the stored history
    hist = " ".join(t.content for t in agent.store.get("p").history)
    assert "4111 1111 1111 1111" not in hist


async def test_interruption_stops_the_agent():
    agent = VoiceAgent(Settings())
    res = await run_call_with_interrupt(
        agent, "i", "Can I book Wednesday at 11am?", interrupt_after_tokens=2
    )
    assert any(e.type == "interrupted" for e in res.events)
