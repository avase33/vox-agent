#!/usr/bin/env python3
"""Run a full appointment-booking call offline and save the agent's voice to WAV.

    python examples/booking_call.py

Prints the transcript + latency and writes ``agent_reply.wav`` so you can listen
to the (synthetic) audio the pipeline actually produced.
"""

from __future__ import annotations

import asyncio

from vox_agent.audio.synth import wav_bytes
from vox_agent.pipeline import VoiceAgent
from vox_agent.sim import SCRIPTS, run_call


async def main() -> None:
    agent = VoiceAgent()
    res = await run_call(agent, "example", SCRIPTS["booking"])

    print("\n--- transcript ---")
    for line in res.transcript_lines:
        print(" ", line)

    print("\n--- tool calls ---")
    for t in res.tool_calls():
        print(f"  {t['name']}({t['arguments']})")

    audio = b"".join(e.audio for e in res.events if e.type == "audio")
    with open("agent_reply.wav", "wb") as f:
        f.write(wav_bytes(audio, agent.settings.sample_rate))
    print(f"\nwrote agent_reply.wav ({len(audio):,} PCM bytes)")
    print("latency:", agent.metrics.summary())


if __name__ == "__main__":
    asyncio.run(main())
