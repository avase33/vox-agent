# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-07-16

Initial release.

### Added
- Streaming voice pipeline: VAD-endpointed STT → LLM (streaming + function
  calling) → clause-buffered TTS, with sub-500ms offline time-to-first-byte.
- **Barge-in interruption**: per-session flag halts token generation and flushes
  the TTS buffer the moment the caller starts speaking.
- **Offline-first providers** with pluggable real adapters: mock/Deepgram (STT),
  mock/Groq/OpenAI (LLM), mock/Cartesia/ElevenLabs (TTS). Zero API keys required.
- **Function calling** against a SQLite appointment book (`check_availability`,
  `create_booking`) with double-booking protection.
- **Slot-filling dialog state machine** (receptionist) that tracks day/time/name.
- **PII redaction** of cards, phones, SSNs and emails before text reaches the LLM,
  retaining last-4 for confirmation.
- **Context pruning** — recent turns kept verbatim, older history summarised.
- From-scratch audio utilities: energy VAD, PCM/mu-law codecs, resampling, and a
  client-side jitter buffer. No numpy.
- FastAPI full-duplex WebSocket server (`/ws/voice`) + single-file Web Audio
  browser client with on-device speech recognition and barge-in.
- CLI (`vox demo|chat|bench|serve|interrupt-demo`), latency metrics (p50/p95),
  simulation harness, pytest suite, and offline verifiers (`verify_tiny`,
  `verify_full`).
- Docker, docker-compose (with optional Redis), GitHub Actions CI, Makefile.
