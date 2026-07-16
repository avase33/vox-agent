# Architecture

vox-agent is three overlapping asynchronous loops joined by a full-duplex
WebSocket. The design goal is **time-to-first-byte (TTFB)** — the gap between the
caller finishing and the first audio playing back — kept under ~500ms.

```
 caller mic ──▶ WebSocket ──▶ STT ──▶ PII redact ──▶ LLM ──┐
                                                           │ (streams tokens
                                                           │  and tool calls)
                                                           ▼
                                    ToolRegistry ◀──── function call
                                    (SQLite calendar)      │
                                                           ▼
 caller spkr ◀── WebSocket ◀── TTS ◀── clause buffer ◀── tokens
```

## Turn lifecycle

1. **Hear.** Mic audio arrives as PCM16 frames. An energy **VAD**
   (`audio/vad.py`) tracks speech and fires an *endpoint* after a configurable
   silence hangover. The utterance audio goes to the **STT** provider, which
   streams interim transcripts then a final.
2. **Redact.** The final transcript passes through `dialog/pii.py`, which swaps
   card/phone/SSN/email spans for typed placeholders (keeping the last four
   digits in a local vault) *before* any text reaches the LLM or logs.
3. **Think + act.** The redacted turn is appended to `ConversationState` and the
   **LLM** streams a reply. When it emits a **tool call**, the pipeline runs it
   against the `ToolRegistry` (a SQLite appointment book), appends the result to
   history, and re-invokes the LLM to continue — classic function-calling loop.
4. **Speak.** Tokens are buffered into clauses (`pipeline.py::_clause_boundary`).
   The moment a clause closes — a comma, a period, or ~12 words — it's handed to
   the **TTS** provider, whose audio frames stream straight back to the caller.
   Playback therefore starts before the sentence is finished.

## Barge-in (interruption)

Every session owns an `asyncio.Event`. The client detects the caller starting to
talk (browser VAD / `onspeechstart`) and sends an interrupt frame; the server
calls `VoiceAgent.interrupt()`, which sets the flag. The response loop checks it
between every token and inside the TTS emit loop, so generation stops and the TTS
buffer is abandoned within one frame — the agent goes quiet almost instantly. The
turn is recorded as `interrupted` in the metrics.

## Streaming = low latency

A naïve "stop and wait" agent pays `STT_full + LLM_full + TTS_full` serially
(3–5s). vox-agent overlaps them:

- STT emits the final as soon as the endpoint fires.
- The LLM streams; the first token starts the clock on speech.
- The first clause is synthesised and played while later clauses are still being
  generated.

`metrics.py` measures each segment (STT, think/TTFT, TTFB, total) and reports
p50/p95 so the numbers are defensible.

## Providers are swappable

`stt/`, `llm/`, `tts/` and `state.py` each expose a `build_*(settings)` factory
that returns a mock or a real adapter based on environment variables. The
pipeline only ever sees the abstract base classes, so going from an offline demo
to Deepgram + Groq + Cartesia + Redis is pure configuration.

| Layer | Offline default | Real adapters |
| --- | --- | --- |
| STT | `MockSTT` (simulated streaming recogniser) | Deepgram |
| LLM | `MockLLM` (policy-driven, streams + tool calls) | Groq, OpenAI |
| TTS | `MockTTS` (real PCM tone synthesis) | Cartesia, ElevenLabs |
| State | `MemoryStore` | Redis |

## Dialog control

The receptionist **state machine** (`dialog/state_machine.py`) holds explicit
slots (day, time, name) and decides, each turn, whether to *say* something or
*call a tool*. This is what stops the agent from forgetting what it still needs or
being derailed — the failure mode of a single free-form prompt. A hosted LLM
expresses the same policy implicitly through function calling; the mock makes it
explicit and fully deterministic, which is what makes the whole system testable.

## Robustness details

- **Jitter buffer** (`audio/jitter.py`) reorders out-of-order frames and holds a
  small cushion so lossy mobile networks don't make the voice choppy.
- **Context pruning** (`dialog/context.py`) keeps the last N turns verbatim and
  compresses older history into a one-line summary so long calls don't slow the
  LLM.
- **mu-law codec** (`audio/pcm.py`) supports G.711 phone audio (Twilio) as well
  as browser PCM16.
