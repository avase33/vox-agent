#!/usr/bin/env python3
"""Smallest end-to-end smoke test: one booking call, offline, zero keys."""

from __future__ import annotations

import asyncio
import time

from vox_agent.pipeline import VoiceAgent
from vox_agent.sim import SCRIPTS, run_call


async def main() -> int:
    print("tiny check: booting vox-agent (all mock providers)...")
    t0 = time.monotonic()
    agent = VoiceAgent()
    res = await run_call(agent, "tiny", SCRIPTS["booking"])
    dt = time.monotonic() - t0

    tools = [t["name"] for t in res.tool_calls()]
    booked = "all set" in " ".join(res.transcript_lines).lower()
    summary = agent.metrics.summary()

    print(f"  heard + answered {summary['turns']} turns")
    print(f"  tool calls: {tools}")
    print(f"  audio synthesised: {res.audio_bytes():,} PCM bytes")
    print(f"  ttfb p50/p95: {summary['ttfb_p50_ms']}/{summary['ttfb_p95_ms']} ms")

    ok = (
        "check_availability" in tools
        and "create_booking" in tools
        and booked
        and res.audio_bytes() > 0
    )
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} (total {dt:.2f}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
