"""FastAPI WebSocket server for real-time, full-duplex voice.

Protocol on ``/ws/voice`` (JSON control frames + binary audio):

Client -> server
    {"type": "say", "text": "..."}   a transcribed user turn (browser Web Speech
                                     API, or the text demo)
    {"type": "interrupt"}            barge-in: stop the agent immediately
    <binary>                         raw PCM16 mic audio (only used when a real
                                     server-side STT is configured)

Server -> client
    {"type": "user_final", "text": ...}
    {"type": "token", "text": ...}         agent words as they generate
    {"type": "tool", "name": ..., ...}     a function call happened
    {"type": "agent_text", "text": ...}    the finished agent sentence
    {"type": "metrics", ...}               per-turn latency
    {"type": "audio_info", "sample_rate": ...}  precedes a burst of audio frames
    <binary>                               raw PCM16 to play

Run with:  ``vox serve``  (requires ``pip install 'vox-agent[server]'``).
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import Settings
from .pipeline import VoiceAgent

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
except ImportError as e:  # pragma: no cover
    raise RuntimeError("Install server extras:  pip install 'vox-agent[server]'") from e

app = FastAPI(title="vox-agent", version="0.1.0")
_settings = Settings()
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "providers": {
                "stt": _settings.stt_provider,
                "llm": _settings.llm_provider,
                "tts": _settings.tts_provider,
            },
        }
    )


@app.get("/")
async def index() -> HTMLResponse:
    idx = _WEB_DIR / "index.html"
    if idx.exists():
        return HTMLResponse(idx.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>vox-agent</h1><p>web/index.html not found.</p>")


async def _pump(agent: VoiceAgent, ws: WebSocket, session_id: str, events) -> None:
    """Forward AgentEvents to the WebSocket as JSON + binary audio frames."""
    async for ev in events:
        if ev.type == "audio":
            await ws.send_json({"type": "audio_info", "sample_rate": ev.data.get("sample_rate", 16000)})
            await ws.send_bytes(ev.audio)
        elif ev.type == "token":
            await ws.send_json({"type": "token", "text": ev.text})
        elif ev.type == "user_final":
            await ws.send_json({"type": "user_final", "text": ev.text})
        elif ev.type == "tool":
            await ws.send_json({"type": "tool", **ev.data})
        elif ev.type == "tool_result":
            await ws.send_json({"type": "tool_result", **ev.data})
        elif ev.type == "redacted":
            await ws.send_json({"type": "redacted", **ev.data})
        elif ev.type == "turn_done":
            await ws.send_json({"type": "agent_text", "text": ev.text})
            if ev.metrics:
                await ws.send_json({"type": "metrics", **ev.metrics})
        elif ev.type == "interrupted":
            await ws.send_json({"type": "interrupted"})
        elif ev.type == "call_end":
            await ws.send_json({"type": "call_end"})


@app.websocket("/ws/voice")
async def voice(ws: WebSocket) -> None:
    await ws.accept()
    agent = VoiceAgent(_settings)
    session_id = ws.headers.get("sec-websocket-key", "ws")[:16] or "ws"
    try:
        await _pump(agent, ws, session_id, agent.open_call(session_id))
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                kind = data.get("type")
                if kind == "interrupt":
                    agent.interrupt(session_id)
                elif kind == "say" and data.get("text"):
                    await _pump(agent, ws, session_id, agent.handle_text(session_id, data["text"]))
            elif msg.get("bytes") is not None:
                # Real STT path: buffer + endpoint would go here. With the mock
                # STT we can't transcribe raw mic audio, so binary frames are
                # accepted but ignored unless a real STT provider is configured.
                pass
    except WebSocketDisconnect:
        pass
    finally:
        agent.store.drop(session_id)
