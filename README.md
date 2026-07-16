# vox-agent 🎙️

**An autonomous real-time voice agent** — the caller speaks, the agent hears,
thinks, *acts*, and speaks back in well under half a second. It books
appointments, handles interruptions (talk over it and it stops), redacts spoken
credit-card and phone numbers, and drives the whole conversation with a
slot-filling state machine.

The headline feature: **it runs end-to-end with zero API keys.** Every provider —
speech-to-text, the LLM brain, and text-to-speech — ships with an offline mock so
you can `pip install` and immediately hear a full call. Flip three environment
variables to swap in Deepgram, Groq, and Cartesia for production.

```
[ mic ] → STT → PII redact → LLM (streaming + function calling) → TTS → [ speaker ]
                                     │
                                     └─▶ SQLite calendar (check_availability, create_booking)
```

## Why this exists

Text chat is easy; **streaming voice is hard.** You can't wait for the caller to
finish, transcribe the whole clip, wait for the LLM, then synthesise audio — that
feels robotic. vox-agent overlaps all three loops and starts *speaking the first
clause before the LLM has finished the sentence*, which is what keeps latency
human.

## Quickstart

```bash
pip install -e .            # zero third-party deps for the core
vox demo                    # scripted booking call, prints transcript + latency
vox chat                    # type as the caller, talk to the brain
vox interrupt-demo          # watch barge-in cut the agent off mid-sentence
python scripts/verify_full.py
```

Run the browser demo (adds FastAPI + a WebSocket server):

```bash
pip install -e ".[server]"
vox serve                   # http://localhost:8000  → click "Start call"
```

The web client uses your browser's on-device speech recognition for STT and plays
the agent's audio through the Web Audio API, so it's a real, talk-to-it demo with
no cloud STT required.

## What `vox demo` prints

```
====================================================================
vox-agent demo — provider stack: STT=mock LLM=mock TTS=mock
====================================================================
   agent | Thanks for calling Vox Clinic, this is the automated assistant. Would you like to book an appointment?
  caller | Hi, I'd like to book an appointment.
   agent | Sure — what day would you like to come in?
  caller | Wednesday works for me.
   agent | On 2026-07-22 I have 09:00, 11:00, 13:00, 15:00. Which time works best?
  caller | Let's do 11am.
  caller | My name is Jordan Lee.
   agent | I'll book Jordan Lee for wednesday at 11am. Shall I confirm?
  caller | Yes, please book it.
   agent | You're all set for 2026-07-22 at 11:00. Anything else?
  caller | No, that's all. Thank you!
   agent | Thanks for calling. Have a great day!
--------------------------------------------------------------------
  tool calls : 2
             - check_availability({'day': 'wednesday'})
             - create_booking({'day': 'wednesday', 'time': '11am', 'name': 'Jordan Lee', 'reason': 'appointment'})
  latency    : {'turns': 6, 'ttfb_p50_ms': ..., 'ttfb_p95_ms': ..., ...}
```

## Going live (real providers)

Everything is env-driven. Install the extras you want and set the keys:

```bash
pip install -e ".[server,deepgram,groq,cartesia]"

export VOX_STT=deepgram   DEEPGRAM_API_KEY=...
export VOX_LLM=groq        GROQ_API_KEY=...        VOX_LLM_MODEL=llama-3.1-8b-instant
export VOX_TTS=cartesia    CARTESIA_API_KEY=...
export VOX_STATE=redis      REDIS_URL=redis://localhost:6379/0
vox serve
```

No code changes — the factories in `stt/`, `llm/`, `tts/` and `state.py` pick the
adapter from the environment.

## The engineering, feature by feature

| Concern | Where | What it does |
| --- | --- | --- |
| **Streaming pipeline** | `pipeline.py` | Async orchestrator; speaks the first clause before the sentence is done. |
| **Barge-in** | `pipeline.py` + `audio/vad.py` | Per-session interrupt flag halts the LLM and flushes the TTS buffer instantly. |
| **Function calling** | `llm/`, `tools/` | LLM emits a tool call → SQLite calendar runs it → result injected back into context. |
| **State machine** | `dialog/state_machine.py` | Tracks day/time/name slots and steers the call until it can act. |
| **PII redaction** | `dialog/pii.py` | Scrubs cards/phones/SSNs/emails before the LLM sees them; keeps last-4. |
| **Context pruning** | `dialog/context.py` | Keeps recent turns verbatim, summarises the rest to keep the model fast. |
| **Jitter buffer** | `audio/jitter.py` | Reorders / cushions audio frames for smooth playback on lossy networks. |
| **VAD + codecs** | `audio/` | Energy VAD, PCM↔mu-law (Twilio), resampling — all from scratch, no numpy. |
| **Latency metrics** | `metrics.py` | STT / think / TTFB / total, reported as p50 & p95. |

## How "offline" works honestly

A real acoustic model can't run with zero dependencies, so the **mock STT** is a
deterministic streaming-recogniser *simulator*: the simulated caller
(`sim.py`) provides the ground-truth text alongside synthetic-but-real PCM audio,
and the mock streams it back interim-then-final exactly as Deepgram would. The
VAD, codecs, jitter buffer, LLM control flow, tool calls, TTS byte stream and
latency accounting are all genuinely exercised. Swap `VOX_STT=deepgram` and the
same pipeline transcribes real speech.

## Testing

```bash
pip install -e ".[dev]"
pytest -q                     # unit + async pipeline tests
python scripts/verify_full.py # full subsystem report
```

## Layout

```
vox_agent/
  audio/      VAD, PCM/mu-law codecs, resampling, jitter buffer, tone synthesis
  stt/        base + mock + deepgram
  llm/        base + mock (policy-driven) + groq/openai (OpenAI-compatible)
  tts/        base + mock (real PCM) + cartesia + elevenlabs
  tools/      SQLite calendar + function-call registry/schemas
  dialog/     nlu, pii redaction, context pruning, receptionist state machine
  pipeline.py orchestrator (streaming + barge-in + metrics)
  server.py   FastAPI WebSocket server        cli.py   command line
  sim.py      offline call simulation         state.py memory/redis session store
web/          single-file Web Audio + WebSocket client
scripts/      verify_tiny.py, verify_full.py
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the data-flow deep dive.

## License

MIT © 2026 Akhil Vase
