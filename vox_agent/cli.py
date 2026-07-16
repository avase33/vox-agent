"""Command-line entry point: ``vox <command>``.

Commands
--------
demo    Run a scripted appointment-booking call end-to-end and print the
        transcript, the tool calls it made, and the latency summary.
chat    Interactive text REPL — type what the "caller" says, hear (see) the
        agent's replies. Same brain/pipeline, STT skipped.
bench   Run several calls and report p50/p95 latency.
serve   Launch the FastAPI WebSocket server (needs the [server] extra).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .config import Settings
from .pipeline import VoiceAgent
from .sim import SCRIPTS, run_call, run_call_with_interrupt


def _banner(title: str) -> None:
    print("=" * 68)
    print(title)
    print("=" * 68)


async def _demo(scenario: str) -> None:
    agent = VoiceAgent(Settings())
    _banner(f"vox-agent demo — provider stack: "
            f"STT={agent.stt.name} LLM={agent.llm.name} TTS={agent.tts.name}")
    result = await run_call(agent, "demo", SCRIPTS.get(scenario, SCRIPTS["booking"]))
    for line in result.transcript_lines:
        who, _, text = line.partition(": ")
        print(f"  {who:>6} | {text}")
    print("-" * 68)
    tools = result.tool_calls()
    print(f"  tool calls : {len(tools)}")
    for t in tools:
        print(f"             - {t['name']}({t['arguments']})")
    print(f"  audio out  : {result.audio_bytes():,} PCM bytes")
    print("-" * 68)
    print("  latency    :", agent.metrics.summary())


async def _chat() -> None:
    agent = VoiceAgent(Settings())
    _banner("vox-agent chat (type 'quit' to exit)")
    async for ev in agent.open_call("chat"):
        if ev.type == "turn_done" and ev.text:
            print(f"  agent | {ev.text}")
    loop = asyncio.get_event_loop()
    while True:
        try:
            line = await loop.run_in_executor(None, lambda: input("  you   > "))
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip().lower() in ("quit", "exit"):
            break
        ended = False
        async for ev in agent.handle_text("chat", line):
            if ev.type == "turn_done" and ev.text:
                print(f"  agent | {ev.text}")
            if ev.type == "call_end":
                ended = True
        if ended:
            print("  (call ended)")
            break


async def _bench(n: int) -> None:
    agent = VoiceAgent(Settings())
    for i in range(n):
        await run_call(agent, f"bench-{i}", SCRIPTS["booking"])
    _banner("vox-agent bench")
    print("  ", agent.metrics.summary())


def _serve(host: str, port: int) -> None:
    try:
        import uvicorn  # type: ignore
    except ImportError:
        print("Install server extras first:  pip install 'vox-agent[server]'", file=sys.stderr)
        raise SystemExit(1)
    uvicorn.run("vox_agent.server:app", host=host, port=port, log_level="info")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vox", description="Autonomous real-time voice agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run a scripted call")
    d.add_argument("--scenario", default="booking", choices=list(SCRIPTS))

    sub.add_parser("chat", help="interactive text REPL")

    b = sub.add_parser("bench", help="latency benchmark")
    b.add_argument("-n", type=int, default=5)

    s = sub.add_parser("serve", help="run the WebSocket server")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)

    sub.add_parser("interrupt-demo", help="demonstrate barge-in interruption")

    args = p.parse_args(argv)

    if args.cmd == "demo":
        asyncio.run(_demo(args.scenario))
    elif args.cmd == "chat":
        asyncio.run(_chat())
    elif args.cmd == "bench":
        asyncio.run(_bench(args.n))
    elif args.cmd == "serve":
        _serve(args.host, args.port)
    elif args.cmd == "interrupt-demo":
        asyncio.run(_interrupt_demo())
    return 0


async def _interrupt_demo() -> None:
    agent = VoiceAgent(Settings())
    _banner("vox-agent barge-in demo")
    res = await run_call_with_interrupt(agent, "int", "Can I book Wednesday at 11am?")
    interrupted = any(e.type == "interrupted" for e in res.events)
    spoke = sum(1 for e in res.events if e.type == "audio")
    print(f"  agent started speaking ({spoke} audio frames) then caller barged in.")
    print(f"  interruption handled: {interrupted}")


if __name__ == "__main__":
    raise SystemExit(main())
